import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import math
import requests
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


# Dictionary of PortWatch codes
locode = pd.read_excel("GeminiTracker Analysis.xlsx", sheet_name="LOCODE")

portwatch_codes = {}

for _, row in locode.iterrows():
    portwatch_codes[row["LOCODE"]] = row["PortWatch Code"]

#FIXED PARAMETERS
TODAY =  pd.Timestamp.today().normalize()
START = pd.to_datetime("01-01-2022", dayfirst=True)
END = pd.to_datetime("01-01-2025", dayfirst=True)
PORT = "CNNGB"
TIMEFRAME = 'W' # resample daily time series to weekly
forecast_steps = 90
feature = "import" #import volume determines level of congestion.
season = 12

data = pd.read_csv("Daily_Port_Activity_Data_and_Trade_Estimates.csv")
print(data.head())
#print(data.describe())
data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)



time_series = data[(data["portid"] == portwatch_codes[PORT]) & (data["date"] >= START) & (data["date"] <= END)][["date",feature]]
sha_2025 = data[(data["portid"] == portwatch_codes[PORT]) & (data["date"] >= END)][["date",feature]]



port_closure = time_series.copy(deep=True)
port_closure = port_closure.sort_values('date')
## TODO: Flag out dates where day-on-day percentage change in import volume fell by more than 75%.
# Calculate day-on-day percentage change
port_closure['pct_change'] = port_closure['import'].pct_change() * 100  # Multiply by 100 to get percentage
# Flag dates where the day-on-day decrease is greater than 75%
port_closure['flag'] = port_closure['pct_change'] < -75
# Filter rows where the flag is True
flagged_dates = port_closure[port_closure['flag']]
print(flagged_dates)

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot the import volumes over time
plt.plot(port_closure['date'], port_closure['import'], marker='o', label='Import Volume', color='blue')

# Highlight flagged dates
plt.scatter(flagged_dates['date'], flagged_dates['import'], color='red', label='Decrease > 75%', s=100, zorder=5)

# Annotate flagged dates with the percentage decrease
for index, row in flagged_dates.iterrows():
    plt.annotate(
        f'{row["pct_change"]:.2f}%',  # Text to display (percentage decrease)
        (row['date'], row['import']),  # Coordinates for the annotation
        textcoords="offset points",  # Offset the text from the point
        xytext=(0, 10),  # Distance from the point
        ha='center',  # Horizontal alignment
        color='red'  # Text color
    )

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Import Volume')
plt.title(f'Import Volume Time Series for {PORT} with Flagged Dates')
plt.legend()

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.7)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()

## Visualise
# fig, ax = plt.subplots()
# interval = pd.Timedelta(days=1)
# ax.bar(port_import["date"],port_import[feature],width=interval)
# fig.autofmt_xdate()
# plt.show()
time_series.set_index("date",inplace=True)
time_series.sort_index(inplace=True)
#time_series = time_series.resample(TIMEFRAME, label='right', closed='right').mean()

sha_2025.set_index("date",inplace=True)
sha_2025.sort_index(inplace=True)
#sha_2025 = sha_2025.resample(TIMEFRAME,label='right',closed='right').mean()
## TODO: Use daily forecasted import volumes as another factor to predict delay.
print("Fitting SARIMA Model...")
auto_model = auto_arima(time_series, start_P = 1, D = 1, start_Q = 1, m = season, seasonal=True, trace=False,stepwise=True)
print(auto_model.summary())

# Forecast for the next periods (adjust steps as needed)
forecast_days = pd.date_range(start= END, periods=forecast_steps, freq='D')

forecast, conf_int = auto_model.predict(n_periods=forecast_steps, return_conf_int=True)

# Convert forecast to DataFrame for clarity

sarima_forecast_df = pd.DataFrame({
    'Forecast': forecast,
    'Lower CI': conf_int[:, 0],
    'Upper CI': conf_int[:, 1]
})

sarima_forecast_df.index = forecast_days
sarima_forecast_df["ISO Week"] = sarima_forecast_df.index.isocalendar().week

print("SARIMA Forecast")
print(sarima_forecast_df)  # Check the forecast values


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
plt.title('SARIMA Forecast of ' + PORT + ' ' + feature + ' using auto_arima')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%G-%V'))
plt.xticks(rotation=90)
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend(loc='upper left')
plt.grid()
plt.show()


### Backfill training data with import volumes
# train_data = pd.read_excel("GeminiTracker Analysis.xlsx",sheet_name="Training Data",header=0)

# port_cst_data = train_data[train_data["Import Volume"] == 0][["Port", "CST ETB"]]

# port_cst_data["CST ETB"] = pd.to_datetime(port_cst_data["CST ETB"])
# port_cst_data["Import Volume"] = None

# for id, row in port_cst_data.iterrows():
#     port = row["Port"]
#     time_series = data[(data["portid"] == portwatch_codes[port]) & (data["date"] >= START)][["date",feature]]
#     time_series['date'] = pd.to_datetime(time_series['date'])
#     time_series.set_index("date",inplace=True)
#     time_series.sort_index(inplace=True)
#     last_day = time_series.index[-1]

#     cur_etb = row["CST ETB"]

#     if cur_etb <= last_day:
#         print(time_series.loc[cur_etb.date().strftime("%Y-%m-%d")][feature])
#         port_cst_data.loc[id]['Import Volume'] = time_series.loc[cur_etb.date().strftime("%Y-%m-%d")][feature]
#     else:
#         forecast_days_steps = math.ceil((cur_etb-last_day) / pd.Timedelta(days=1))

#         auto_model = auto_arima(time_series, start_P = 1, D = 1, start_Q = 1, m = season, seasonal=True, trace=False,stepwise=True)
#         # Forecast for the next periods (adjust steps as needed)
#         forecast_days = pd.date_range(start= last_day, periods=forecast_days_steps, freq='D')

#         import_forecast = auto_model.predict(n_periods=forecast_days_steps, return_conf_int=False)
#         # Convert forecast to DataFrame for clarity
#         import_forecast_df = pd.DataFrame({'Import Forecast': import_forecast})
#         import_forecast_df.index = forecast_days
#         print(import_forecast_df)
#         port_cst_data.loc[id]['Import Volume'] = import_forecast_df["Import Forecast"].iloc[-1]

# with pd.ExcelWriter("GeminiTracker Analysis.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
#     # Write the DataFrame to a new or existing sheet
#     port_cst_data.to_excel(writer, sheet_name='Import Volumes', index=False)

# print(f"DataFrame written to 'Import Volumes' sheet in the existing workbook.")