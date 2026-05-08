######################################################
#Packages
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense



import random
import numpy as np
import tensorflow as tf

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})
# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
######################################################
#Data Setup
deaths = pd.read_excel('cmi2019final.xlsx', sheet_name='EW Male deaths', header=1, index_col=0)
popest = pd.read_excel('cmi2019final.xlsx', sheet_name='EW Male pop', header=1, index_col=0)

deaths = deaths.loc[60:89]
popest = popest.loc[60:89]

mx = deaths / popest
qx = mx / (1 + 0.5 * mx)
qx = qx.clip(lower=1e-10, upper=1 - 1e-10)

logit_qx = np.log(qx / (1 - qx))
######################################################
# Least Squares
ages = logit_qx.index.astype(float).values
x_bar = np.mean(ages)
age_centered = ages - x_bar

k1_list = []
k2_list = []

for year in logit_qx.columns:
    y = logit_qx[year].values
    X = np.column_stack([np.ones(len(ages)), age_centered])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    k1_list.append(coeffs[0])
    k2_list.append(coeffs[1])

k1 = pd.Series(k1_list, index=logit_qx.columns)
k2 = pd.Series(k2_list, index=logit_qx.columns)
######################################################
#Train Test Split
split_year = 2007

k2_train = k2.loc[:split_year]
k2_test = k2.loc[split_year+1:]
k1_train = k1.loc[:split_year]
k1_test = k1.loc[split_year+1:]
######################################################
#ARIMA
arima_model = ARIMA(k2_train, order=(0, 1, 0))
arima_fitted = arima_model.fit()

n_forecast = len(k2_test)
forecast = arima_fitted.forecast(steps=n_forecast)
ARIMA1 = pd.Series(forecast.values, index=k2_test.index)

print("k2 forecast:", ARIMA1.values[:5])
print("k2 actual:", k2_test.values[:5])

rmse_arima = np.sqrt(mean_squared_error(k2_test, ARIMA1))
mae_arima = mean_absolute_error(k2_test, ARIMA1)
print("RMSE:", rmse_arima)
print("MAE:", mae_arima)
######################################################
#LSTM
scaler = StandardScaler()
k2_scaled = scaler.fit_transform(k2_train.values.reshape(-1, 1))

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), 0])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

window_size = 10
X_train, y_train = create_sequences(k2_scaled, window_size)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print("Shape of X_train:", X_train.shape)

model = Sequential([
    LSTM(50, activation='tanh', input_shape=(window_size, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.1,
    verbose=1
)
######################################################
#LSTM Forecasting
last_window = k2_scaled[-window_size:].reshape(1, window_size, 1)

k2_forecast_scaled = []
for _ in range(32):
    pred = model.predict(last_window, verbose=0)
    k2_forecast_scaled.append(pred[0, 0])
    last_window = np.append(last_window[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

k2_lstm_forecast = scaler.inverse_transform(np.array(k2_forecast_scaled).reshape(-1, 1))
full_index = pd.date_range(start='2008-01-01', periods=32, freq = 'YS')
k2_lstm_forecast = pd.Series(k2_lstm_forecast.flatten(), index=full_index)

print("LSTM forecast:", k2_lstm_forecast.values[:5])
print("Actual k2:", k2_test.values[:5])

k2lstm = k2_lstm_forecast[:len(k2_test)]

rmse_lstm = np.sqrt(mean_squared_error(k2_test, k2lstm))
mae_lstm = mean_absolute_error(k2_test, k2lstm)
print("RMSE:", rmse_lstm)
print("MAE:", mae_lstm)
######################################################
#ARIMA for k1
k1_train_arima = k1_train.copy()
k1_train_arima.index = pd.date_range(start=str(k1_train.index[0]), periods=len(k1_train), freq='YS')

arima_model = ARIMA(k1_train_arima, order=(0, 1, 0))
arima_fitted = arima_model.fit()

n_forecast = 32
forecast = arima_fitted.forecast(steps=n_forecast)
indexk1 = pd.date_range(start='2008-01-01', periods=32, freq = 'YS')
ARIMA2 = pd.Series(forecast.values, index=indexk1)

print("k1 forecast:", ARIMA2.values[:5])
print("k1 actual:", k1_test.values[:5])
######################################################
#Premium Calculations
logit_qx_forecast = pd.DataFrame(
    np.outer(age_centered, k2_lstm_forecast.values) + ARIMA2.values,
    index=logit_qx.index,      # ages 60-89
    columns=indexk1  # test years 2008-2039
)

logit_qx_forecast.columns=logit_qx_forecast.columns.year




insurance_type=0
#insurance_type = int(input("Enter 0 to skip, 1 for Term life Insurance, 2 for Pure Endowment Insurance or 3 for Endowment Insurance: "))
if insurance_type==1 or insurance_type==2 or insurance_type==3:
    age_input = int(input("Enter age: "))
    year_input = int(input("Enter policy start year (2008-2019): "))
    term_input = int(input("Enter term (years): "))
    rate_input = float(input("Enter interest rate (e.g. 0.05 for 5%): "))
    benefit_input = float(input("Enter sum assured (e.g. 50000): "))





def term_life_premium(age, forecast_start_year, term, interest_rate, benefit):
    apv_benefit = 0
    apv_premiums = 0
    survival_prob = 1.0

    for t in range(1, term + 1):
        current_age = age + t - 1
        current_year = forecast_start_year + t - 1

        if current_age not in logit_qx_forecast.index:
            print(f"Age {current_age} out of range (must be 60-89)")
            break
        if current_year not in logit_qx_forecast.columns:
            print(f"Year {current_year} out of forecast range (must be 2008-2039)")
            break

        logit_val = logit_qx_forecast.loc[current_age, current_year]
        qx = np.exp(logit_val) / (1 + np.exp(logit_val))
        px = 1 - qx

        apv_premiums += survival_prob / (1 + interest_rate) ** (t - 1)
        apv_benefit += survival_prob * qx * benefit / (1 + interest_rate) ** t

        survival_prob *= px

    if apv_premiums == 0:
        return 0, 0, 0

    premium = apv_benefit / apv_premiums
    return premium, apv_benefit, apv_premiums

def pure_endowment(age,forecast_start_year,term,interest_rate,benefit):
    apv_benefit = 0
    apv_premiums = 0
    survival_prob = 1.0
    for t in range(1, term + 1):
        current_age = age + t - 1
        current_year = forecast_start_year + t - 1
        if current_age not in logit_qx_forecast.index or current_year not in logit_qx_forecast.columns:
            print(f"Year {current_year} or Age {current_age} out of range")
            break
        apv_premiums += survival_prob / (1 + interest_rate) ** (t - 1)

        logit_val = logit_qx_forecast.loc[current_age, current_year]
        qx = np.exp(logit_val) / (1 + np.exp(logit_val))
        px = 1 - qx
        survival_prob *= px
    apv_benefit = (survival_prob * benefit) / (1 + interest_rate) ** term
    if apv_premiums == 0:
        return 0, 0, 0

    premium = apv_benefit / apv_premiums
    return premium, apv_benefit, apv_premiums

def endowment(age, forecast_start_year,term,interest_rate,benefit):
    apv_benefit = 0
    apv_premiums = 0
    survival_prob = 1.0
    apv_pure_endowment = 0
    total_benefit_apv = 0
    for t in range(1, term + 1):
        current_age = age + t - 1
        current_year = forecast_start_year + t - 1

        if current_age not in logit_qx_forecast.index or current_year not in logit_qx_forecast.columns:
            print(f"Year {current_year} or Age {current_age} out of range")
            break
        apv_premiums += survival_prob / (1 + interest_rate) ** (t - 1)

        logit_val = logit_qx_forecast.loc[current_age, current_year]
        qx = np.exp(logit_val) / (1 + np.exp(logit_val))

        apv_benefit += (survival_prob * qx * benefit) / (1 + interest_rate) ** t

        survival_prob *= (1-qx)
    apv_pure_endowment = (survival_prob * benefit) / (1 + interest_rate) ** term

    total_benefit_apv = apv_benefit + apv_pure_endowment
    premium = total_benefit_apv / apv_premiums
    return premium, total_benefit_apv, apv_premiums



if insurance_type==1:
    premium, apv_b, apv_p = term_life_premium(age_input, year_input, term_input, rate_input, benefit_input)
    print(f"\n--- Term Life Insurance ---")
elif insurance_type==2:
    premium, apv_b, apv_p = pure_endowment(age_input, year_input, term_input, rate_input, benefit_input)
    print(f"\n--- Pure Endowment Insurance ---")
elif insurance_type==3:
    premium, apv_b, apv_p = endowment(age_input, year_input, term_input, rate_input, benefit_input)
    print(f"\n--- Endowment Insurance ---")

if insurance_type != 0:
    print(f"Age: {age_input}, Start Year: {year_input}, Term: {term_input} years")
    print(f"Interest Rate: {rate_input*100}%, Sum Assured: £{benefit_input:,.2f}")
    print(f"APV of Benefit:   £{apv_b:,.4f}")
    print(f"APV of Premiums:  £{apv_p:,.4f}")
    print(f"Annual Premium:   £{premium:,.4f}")
##################################################
#Cohort testing
fitted = pd.DataFrame(
    np.outer(age_centered, k2.values) + k1.values,
    index=logit_qx.index,
    columns=logit_qx.columns
)
residuals_check = logit_qx - fitted

#plt.figure(figsize=(10, 6))
#plt.imshow(residuals_check.values, aspect='auto', cmap='RdBu')
#plt.colorbar()
#plt.title('CBD Residuals')
#plt.xlabel('Year')
#plt.ylabel('Age')
#plt.show()

#plt.figure(figsize=(12, 6))
#plt.plot(k2.index, k2.values, label='Observed k²_t', color='black')
#plt.plot(k2_test.index, ARIMA1.values, label='ARIMA Forecast', color='blue', linestyle='--')
#plt.plot(k2_test.index, k2lstm.values, label='LSTM Forecast', color='red', linestyle='--')
#plt.axvline(x=split_year, color='green', linestyle=':', label='Train/Test Split')
#plt.legend()
#plt.title('Males k²t ARIMA and LSTM Forecasts vs Observed')
#plt.xlabel('Year')
#plt.ylabel('k²_t')
#plt.show()

import pandas as pd
import matplotlib.pyplot as plt


results = {
    'Model': ['LSTM'],
    'MAE': [mae_lstm],
    'RMSE': [rmse_lstm]
}
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

#plt.figure(figsize=(12, 6))
#plt.plot(k2_test.index, k2_test.values, label='Observed k²_t', color='black')
#plt.plot(k2_test.index, ARIMA1.values, label='ARIMA Forecast', color='blue', linestyle='--')
#plt.plot(k2_test.index, k2lstm.values, label='LSTM Forecast', color='red', linestyle='--')
#plt.legend()
#plt.title('Male k²_t Observed vs Forecasted (2008-2019)')
#plt.xlabel('Year')
#plt.ylabel('k²_t')
#plt.show()
