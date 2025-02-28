import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import math
import itertools
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import ssl
from pmdarima import auto_arima

warnings.filterwarnings('ignore')

# Create default SSL context to bypass SSL Certificate Verification Issues
ssl._create_default_https_context = ssl._create_unverified_context

#FIXED PARAMETERS
INPUT_PORT = "Tianjin"
START = pd.to_datetime("01-01-2024")
END = pd.to_datetime("31-01-2025")
forecast_steps = 10
feature = "Median Waiting Time Off Port Limits"
season = 12


data = pd.read_excel("GeminiTracker Analysis.xlsx",sheet_name="Port Congestion",header=0)
print(data.head())
#Flash-fill NA values 
data["Median Waiting Time Off Port Limits"] = data["Median Waiting Time Off Port Limits"].fillna(method="ffill")
data["Median Time At Port"] = data["Median Time At Port"].fillna(method="ffill")

# convert to hours
data["Median Waiting Time Off Port Limits"] = data["Median Waiting Time Off Port Limits"] * 24
data["Median Time At Port"] = data["Median Time At Port"] * 24

# Convert Year + Week Number to a Date
dates = data.Year.astype(str) + "-" + data.Week.astype(str)
data["Date"] =  pd.to_datetime(dates + "-1", format='%G-%V-%u')
data.tail()

# See unique ports
ports = data["Port"].unique()
print(ports)

# Select port congestion data for specified port within given date range
shanghai = data[(data["Port"] == INPUT_PORT) & (data["Date"] >= START) & (data["Date"] <= END)]
sha_2025 = data[(data["Port"] == INPUT_PORT) & (data["Date"] > END)]

shanghai.set_index("Date",inplace=True)
sha_2025.set_index("Date",inplace=True)
shanghai.sort_index(inplace=True)
sha_2025.sort_index(inplace=True)

time_series = shanghai[feature]

# Decompose time series into seasonal, trend and residual components
decomposition = sm.tsa.seasonal_decompose(time_series, model='additive', 
                            period=12) #additive or multiplicative is data specific
fig = decomposition.plot()
plt.show()

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(time_series, lags=20, ax=axes[0])
plot_pacf(time_series, lags=20, ax=axes[1])
plt.show()


# Check for stationarity with ADF Test

def adf_test(series):
    result = adfuller(series, autolag="AIC")
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])
    
    if result[1] < 0.05:
        message = "✅ The time series is stationary (reject H₀)."
    else:
        message = "❌ The time series is NOT stationary (fail to reject H₀)."
        
    return message

# Example: Test stationarity on 'Median_Waiting_Time'
print("For original time series, " + adf_test(time_series))

diff_port = time_series.copy()
diff_port = diff_port.diff()
diff_port.dropna(inplace=True)

# Test stationarity on differenced series
print("For differenced time series, " + adf_test(diff_port))

## Fit ARIMA Model, find optimal (p,d,q) parameters based on AIC value
best_aic = float("inf")
best_order = None
best_model = None

for p in range(4):  # Adjust range if needed
    for d in range(3):
        for q in range(4):
            try:
                model = ARIMA(time_series, order=(p, d, q))
                model_fit = model.fit()
                aic = model_fit.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    best_model = model_fit
                print(f'ARIMA{(p,d,q)} - AIC:{aic:.2f}')
            except:
                continue


message = f'\n✅ Best ARIMA{best_order} model - AIC:{best_aic:.2f}'
print(message)


## Plot residuals
best_model.plot_diagnostics(figsize=(15,8))

# Fit ARIMA Model
if best_order[1] == 0 and "NOT" in message:
    best_order = (best_order[0],1,best_order[2])

model = ARIMA(time_series, order = best_order).fit()

# Forecast Next 7 Weeks (2025)
forecast_weeks = pd.date_range(start= END, periods=forecast_steps, freq='W-MON')
forecast = model.get_forecast(steps=forecast_steps)
forecast_values = forecast.predicted_mean
conf_int = forecast.conf_int()

if message.__contains__("NOT"):
    # Get the last actual value before forecasting
    last_actual_value = time_series.iloc[-1]

    # Reconstruct actual forecast values
    forecast_values = forecast_values.cumsum() + last_actual_value
    conf_int.iloc[:,0] = conf_int.iloc[:,0] + forecast_values
    conf_int.iloc[:,1] = conf_int.iloc[:,1] + forecast_values

# Convert Forecast to DataFrame
forecast_df = pd.DataFrame({"Forecast": list(forecast_values)}, index=forecast_weeks)

## Check for autocorrelation in residuals with Ljung-Box Test
ljung_box_res = acorr_ljungbox(model.resid,lags=[26],return_df=True)
if ljung_box_res['lb_pvalue'].values[0] < 0.05:
    message = "Autocorrelation exists in residuals."
else:
    message = "No significiant autocorrelation."
 
print(message)

# Plot
plt.figure(figsize=(14, 6))
plt.plot(shanghai.index, shanghai[feature], label="Historical Data", linestyle="-", color="blue")
plt.plot(forecast_df.index, forecast_df["Forecast"], label="Forecast (2025)", linestyle="-", color="red", marker='o')
plt.plot(sha_2025.index, sha_2025[feature], label="Actual", linestyle = "-", color = "green", marker="o")

#plot lower and upper confidence intervals
plt.fill_between(forecast_df.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color="pink", alpha = 0.3,label="95% Confidence Interval")

# Format X-axis

plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0)) #Monday
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%G-%V'))
plt.xticks(rotation=90)
plt.xlabel("Date")
plt.ylabel("Median Waiting Time (hours)")
plt.title(INPUT_PORT + " Port " + feature + " ARIMA Model Forecast")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


## Evaluate performance of time series model

low = min(len(sha_2025),len(forecast_values))

mae = mean_absolute_percentage_error(sha_2025.loc[forecast_weeks[0]:forecast_weeks[low-1]][feature],forecast_values[0:low])
rmse = math.sqrt(mean_squared_error(sha_2025.loc[forecast_weeks[0]:forecast_weeks[low-1]][feature],forecast_values[0:low]))
print(f"ARIMA MAE: {mae:.2f}")
print(f"ARIMA RMSE: {rmse:.2f}")

# Fit SARIMA Model with auto_arima
auto_model = auto_arima(time_series, start_P = 1, D = 1, start_Q = 1, m = season, seasonal=True, trace=True,stepwise=True)
print(auto_model.summary())

# Forecast for the next periods (adjust steps as needed)

forecast, conf_int = auto_model.predict(n_periods=forecast_steps, return_conf_int=True)
print(forecast)

# Convert forecast to DataFrame for clarity

sarima_forecast_df = pd.DataFrame({
    'Forecast': forecast,
    'Lower CI': conf_int[:, 0],
    'Upper CI': conf_int[:, 1]
})

sarima_forecast_df.index = forecast_weeks

print(sarima_forecast_df.head())  # Check the forecast values

plt.figure(figsize=(12, 6))

# Plot historical data
plt.plot(time_series, label='Observed', color='blue')

# Plot forecast
plt.plot(sarima_forecast_df.index,sarima_forecast_df["Forecast"], label='Forecast', linestyle = '-',marker = 'o', color='red')
plt.plot(sha_2025.index, sha_2025[feature], label="Actual", linestyle = "-", color = "green", marker="o")

# Confidence intervals
plt.fill_between(sarima_forecast_df.index,
                 sarima_forecast_df['Lower CI'],
                 sarima_forecast_df['Upper CI'],
                 color='pink', alpha=0.3, label='Confidence Interval')
plt.title('SARIMA Forecast of ' + INPUT_PORT + ' ' + feature + ' using auto_arima')
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0)) #Monday
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%G-%V'))
plt.xticks(rotation=90)
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend(loc='upper left')
plt.grid()
plt.show()

# Check residuals
auto_model.plot_diagnostics(figsize=(15,8))

sarima_forecast_values = sarima_forecast_df["Forecast"]
low = min(len(sha_2025),len(sarima_forecast_values))

mae = mean_absolute_percentage_error(sha_2025.loc[forecast_weeks[0]:forecast_weeks[low-1]][feature],sarima_forecast_values[0:low])
rmse = math.sqrt(mean_squared_error(sha_2025.loc[forecast_weeks[0]:forecast_weeks[low-1]][feature],sarima_forecast_values[0:low]))
print(f"SARIMA MAPE: {mae:.2f}")
print(f"SARIMA RMSE: {rmse:.2f}")