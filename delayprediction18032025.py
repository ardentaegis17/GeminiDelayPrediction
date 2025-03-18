
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from pmdarima import auto_arima
from meteostat import Point, Daily
import math
import ssl
import warnings
import joblib

warnings.filterwarnings('ignore')

DAYS = 10
# make predictions for past voyages.
START_OFFSET_DAYS = 0

TODAY =  pd.Timestamp.today().normalize() - pd.Timedelta(days = START_OFFSET_DAYS)
NEXT = TODAY + pd.Timedelta(days=DAYS) 
IMPORT_START = pd.to_datetime("01-01-2022", dayfirst=True)
IMPORT_END = pd.to_datetime("01-03-2025", dayfirst=True)
SPEED = 16 #in knots
FEATURE = "Median Waiting Time Off Port Limits" #in anchorage

print(TODAY,NEXT)

# 0. Load in data from GeminiTrackerAnalysis Workbook

# A. GEMINI DASHBOARD
gemini_df = pd.read_excel("GeminiTracker Analysis.xlsx",sheet_name="Dashboard",header=0)
gemini_df.replace('OMIT', 0, regex = True, inplace = True)
gemini_df['Next Arrival Date'] = pd.to_datetime(gemini_df['Next Arrival Date'],dayfirst=True)
gemini_df['Date'] = pd.to_datetime(gemini_df['Date'],dayfirst=True)

#filter the vessels arriving in the period between TODAY and NEXT.
arriving_vessels = gemini_df[(gemini_df["Next Arrival Date"] > TODAY + pd.Timedelta(days=1)) & (gemini_df["Next Arrival Date"] < NEXT + pd.Timedelta(days=1)) & (gemini_df["Date"] == TODAY)]

#print(arriving_vessels[["SSY", VSL NAME", "VOY NO","Next Arrival Date"]])
#print(len(arriving_vessels))

# filter for HL vessels
hl_vessels = arriving_vessels[arriving_vessels["OPR"] == "HL"]
msk_vessels = arriving_vessels[arriving_vessels["OPR"] == "MSK"]
# B. SEA DISTANCES
sea_distances = pd.read_excel("GeminiTracker Analysis.xlsx",sheet_name="Sea Distances",header=0)

# C. PORT CONGESTION
port_congestion = pd.read_excel("GeminiTracker Analysis.xlsx",sheet_name="Port Congestion",header=0)
# flash fill NA values
port_congestion[FEATURE] = port_congestion[FEATURE].fillna(method="ffill")
port_congestion[FEATURE] = port_congestion[FEATURE] * 24 # convert to hours

# Convert Year + Week Number to a Date
dates = port_congestion.Year.astype(str) + "-" + port_congestion.Week.astype(str)
port_congestion["Date"] = pd.to_datetime(dates + "-1", format="%G-%V-%u") # -1 stands for Monday

# D. WEATHER (METEOSTAT)
portwatch_codes = {}
locode = pd.read_excel("GeminiTracker Analysis.xlsx", sheet_name="LOCODE")

print("Extracting today's weather data....")

try:
   weather_df = pd.read_csv("weather" + TODAY.strftime('%d%m%Y') + '.csv')
except:
    weather_df = pd.DataFrame()
    for index, row in locode.iterrows():
    # Create Point for Port
        port = Point(float(row["Latitude"]), float(row["Longitude"]))
        port.max_count = 1 # only obtain data from one weather station.

        # Get weekly data for 2024
        data = Daily(port, TODAY,NEXT)
        try:
            data = data.normalize()
            # data = data.aggregate("W-MON")
            data = data.fetch()
            data.index = pd.to_datetime(data.index,yearfirst=True)
            data = data[data.index.to_series().between(TODAY,NEXT)]
            data["LOCODE"] = row["LOCODE"]
        except KeyError:
            continue
        
        weather_df = pd.concat([weather_df,data])
    weather_df.to_csv('weather' + TODAY.strftime('%d%m%Y') + '.csv')

weather_df = pd.read_csv("weather" + TODAY.strftime('%d%m%Y') + '.csv')
weather_df['time'] = pd.to_datetime(weather_df['time'])

# E. Import Volume Estimates

for _, row in locode.iterrows():
    portwatch_codes[row["LOCODE"]] = row["PortWatch Code"]

import_data = pd.read_csv("Daily_Port_Activity_Data_and_Trade_Estimates.csv")
import_data["date"] = pd.to_datetime(import_data["date"]).dt.tz_localize(None)

raw_data = f"Raw_Data_{TODAY.strftime("%d%m%Y")}"

if raw_data in pd.ExcelFile("GeminiTracker Analysis.xlsx").sheet_names:
    print(f"Raw data for {TODAY.strftime('%d%m%Y')} already exists! Skipping the generation for raw features. ")
    predictions_df = pd.read_excel("GeminiTracker Analysis.xlsx",sheet_name=raw_data,header=0)
else: 
    # Define the Prediction DataFrame columns
    columns = [
        'SSY', 'VOY NO', 'VSL NAME', 'LEG', 'Prev. Port', 'Prev ETB', 
        'Port', 'CST ETB', 'Wind Spd', 'AirPres', 'Fcst Prev Delay', 'Actual Prev Delay', 
        'PortCongestion', 'Import Volume', 'Sea Time (Hrs)', 'Weather Multiplier', 
        'Total Time Allocated', 'Total Time Needed', 'Recorded Delay','Predicted Delay (HOURS)','Total Delay','Date Predicted'
    ]

    predictions_df = pd.DataFrame(columns=columns)
    congestion_cache = {}
    import_cache = {}

    print(f"All data loaded and {len(hl_vessels)} arriving HL vessels in the next {DAYS} days have been extracted. Generating raw features....")


    for _, vsl in hl_vessels.iterrows():

        # 1. Get SSY, VSL NAME, VOY NO, current leg
        ssy = vsl["SSY"]
        name = vsl["VSL NAME"]
        voy_no = vsl["VOY NO"]
        cur_leg = int(vsl["Cur. Leg"])


        ## -----------------------------------------------------

        # 2. Get delay experienced at previous leg
        if cur_leg == 1:
            # We don't have P/O port listed in dashboard. Thus we cannot determine sea time,prev delay, and time allocated. Skip for now.
            print(f"Skipping {name} in {ssy} with voyage number {voy_no} on Leg {cur_leg} due to lack of P/O port info and ETB.")
            continue

        print(f"Predicting delay for {name} in {ssy} with voyage number {voy_no} on Leg {cur_leg}...")


        ## Process for determining previous leg delay at leg X:

        ## a. Take the last record for leg x-1. Take next arrival date at port for leg x-1 and forecasted delay for leg x-1.
        prev_leg_record = gemini_df[(gemini_df["VSL NAME"] == name) & (gemini_df["VOY NO"] == voy_no) & (gemini_df["SSY"] == ssy) & (gemini_df["Cur. Leg"] == cur_leg - 1) & (gemini_df["Date"] < TODAY + pd.Timedelta(days=1))].tail(1)
        if prev_leg_record.empty:
            print(f"Skipping {name} in {ssy} with voyage number {voy_no} on Leg {cur_leg} due to lack of previous leg record.")
            continue
        
        prev_etb = prev_leg_record["Next Arrival Date"].iloc[0]
        prev_delay_forecast = prev_leg_record["d"+str(cur_leg-1)].iloc[0]

        ### b. Take the last available record for leg x. Take the next arrival date at port for leg x actualised delay for leg x-1.
        cur_leg_record = gemini_df[(gemini_df["VSL NAME"] == name) & (gemini_df["VOY NO"] == voy_no) & (gemini_df["SSY"] == ssy) & (gemini_df["Cur. Leg"] == cur_leg) & (gemini_df["Date"] < TODAY + pd.Timedelta(days=1))].tail(1)
        cur_etb = cur_leg_record["Next Arrival Date"].iloc[0]
        prev_delay_actual = cur_leg_record["d"+str(cur_leg-1)].iloc[0]

        ### The prev. leg delay is actualised - forecasted.
        prev_delay = prev_delay_actual - prev_delay_forecast

        ## Total Time allocated = Current ETB - Prev ETB
        ### The prev. ETB will be the next arrival date listed in the last record for leg x-1.
        ### The current ETB is the next arrival date listed in the last available record for leg x.
        time_allocated = (cur_etb - prev_etb)/ pd.Timedelta(hours=1) # convert to hours

        ### Obtain delays already forecasted for leg x.
        cur_delay = cur_leg_record['d'+str(cur_leg)].iloc[0]

        ## ----------------------------------------------------------------

        # 3. Get the sea time between ports.

        start_port = vsl["leg"+str(cur_leg-1)]
        end_port = vsl["leg"+str(cur_leg)]
        
        sea_distance = sea_distances[(sea_distances["KEY"] == start_port+end_port) | (sea_distances["KEY"] == end_port+start_port)]["Distance (nautical miles)"]
        
        if sea_distance.empty:
            print(f"Sea distance not found for ports {start_port} and {end_port}. Setting to 0.")
            sea_distance = 0
        else:
            sea_distance = sea_distance.iloc[0]
            print(f"This vessel is currently travelling between {start_port} and {end_port}, which are {sea_distance} nautical miles apart. Its Coastal ETB is {cur_etb}.")

        sea_time = round(sea_distance / SPEED,1)

        ## -----------------------------------------------------------------------

        # 4. Get port congestion forecast for the week that vessel is going to arrive.
        
        # Obtain the ISO week numbers of today and current ETB.
        week_today = TODAY.isocalendar().week
        week_arrival = cur_etb.isocalendar().week
        forecast_steps = week_arrival - week_today + 1
        

        port_data = port_congestion[port_congestion["Port"] == end_port]
        port_data.set_index("Date", inplace=True)
        port_data.sort_index(inplace=True)

        time_series = port_data[FEATURE]

        # Check if forecast was previously generated.
        if congestion_cache.get((end_port,week_arrival),-1) != -1:
            congestion_hours = congestion_cache[(end_port,week_arrival)]
        elif TODAY > cur_etb: # no need to forecast, port congestion data already present in dataset.
            cur_mon = cur_etb.dt.to_period('W').apply(lambda r: r.start_time) # get the Monday for this week.
            congestion_hours = time_series[cur_mon].iloc[0]
        else: 
            forecast_weeks = pd.date_range(start=TODAY-pd.Timedelta(days=6), periods=forecast_steps, freq='W-MON')

            ## Fit SARIMA Model with auto_arima
            season = 12 # seasonal patterns every quarter (12 weeks)
            auto_model = auto_arima(time_series, start_P = 1, D = 1, start_Q = 1, m = season, seasonal=True,trace=False,stepwise=True)
            forecast = auto_model.predict(n_periods = forecast_steps, return_conf_int = False)

            # Convert forecast to DataFrame for clarity
            sarima_forecast_df = pd.DataFrame({'Congestion Forecast': forecast})
            sarima_forecast_df.index = forecast_weeks
            sarima_forecast_df["ISO Week"] = sarima_forecast_df.index.isocalendar().week

            # Cache the forecasts to avoid re-fitting models for the same ports.

            for _, row in sarima_forecast_df.iterrows():
                wk = row["ISO Week"]
                fcst = row["Congestion Forecast"]
                congestion_cache[(end_port, wk)] = fcst

            print(sarima_forecast_df)

            congestion_hours = sarima_forecast_df[sarima_forecast_df["ISO Week"] == week_arrival]["Congestion Forecast"].iloc[0]
        
        #4B. Get daily import volume forecast for the day vessel is going to arrive
        if import_cache.get((cur_etb.date(),end_port),-1) != -1:
            import_volume = import_cache[(cur_etb.date(),end_port)]
        else:
            import_series = import_data[(import_data["portid"] == portwatch_codes[end_port]) & (import_data["date"] >= IMPORT_START) & (import_data["date"] <= IMPORT_END)][["date","import"]]
            import_series.set_index("date",inplace=True)
            import_series.sort_index(inplace=True)

            last_date = import_series.index[-1]

            forecast_day_steps = math.ceil((cur_etb - last_date) / pd.Timedelta(days=1))

            auto_model = auto_arima(import_series, start_P = 1, D = 1, start_Q = 1, m = season, seasonal=True, trace=False,stepwise=True)
            # Forecast for the next periods (adjust steps as needed)
            forecast_days = pd.date_range(start= IMPORT_END, periods=forecast_day_steps, freq='D')

            import_forecast = auto_model.predict(n_periods=forecast_day_steps, return_conf_int=False)
            # Convert forecast to DataFrame for clarity
            import_forecast_df = pd.DataFrame({'Import Forecast': import_forecast})
            import_forecast_df.index = forecast_days
            print(import_forecast_df)
            import_volume = import_forecast_df["Import Forecast"].iloc[-1]
            import_cache[(cur_etb.date(),end_port)] = import_volume

        ## ----------------------------------------------------------------------------

        # 5. Get weather forecast for the upcoming day of arrival (if available)
        port_weather = weather_df[(weather_df["time"] == (cur_etb.date()).strftime('%Y-%m-%d')) & (weather_df["LOCODE"] == end_port)]

        ## Weather multiplier defaults to 1. High wind speed and low pressure are signs of nearby typhoon.
        ## - If wind speed >= 12, add 0.1. 
        ## - If wind speed >= 15, add 0.3.
        ## - If pressure < 1005, add 0.5. 

        
        weather_mult = 1
        if port_weather.empty:
            print(f"Weather data could not be obtained for {end_port} on {cur_etb.date().strftime('%Y-%m-%d')}.")
            port_wind = ''
            port_pres = ''
        else:
            # Extract wind speed and air pressure.
            port_wind = port_weather["wspd"]
            port_pres = port_weather["pres"]

        
            if not port_wind.empty:
                port_wind = port_wind.iloc[0]
                if port_wind >= 12:
                    weather_mult += 0.1

                if port_wind >= 15:
                    weather_mult += 0.3
            else:
                port_wind = -1

            if not port_pres.empty:
                port_pres = port_pres.iloc[0]
                if port_pres <= 1005:
                    weather_mult += 0.5
            else:
                port_pres = 9999


        ## --------------------------------------------------------------------------------
        # 6. BEFORE: Apply rules-based algo to determine expected delay.
        ## Predicted Delay = Time Allocated - Est. Time Needed
        ## Est. Time Needed = Prev Leg Delay + (Forecasted Port Congestion + Sea Time) * Weather_Multiplier

        ## NOW: Store all the information as raw data.

        est_time = (prev_delay + congestion_hours) * weather_mult + sea_time
        predicted_delay = round(est_time - time_allocated,1)
        predict_row = {
        'SSY': ssy,
        'VOY NO': voy_no,
        'VSL NAME': name,
        'LEG': cur_leg,
        'Prev. Port': start_port,
        'Prev ETB': prev_etb,
        'Port': end_port,
        'CST ETB': cur_etb,
        'Wind Spd': port_wind,
        'AirPres': port_pres,
        'Fcst Prev Delay': prev_delay_forecast,
        'Actual Prev Delay': prev_delay_actual,
        'PortCongestion': congestion_hours,
        'Import Volume': import_volume,
        'Sea Time (Hrs)': sea_time,
        'Weather Multiplier': weather_mult,
        'Total Time Allocated': time_allocated,
        'Total Time Needed': est_time,
        'Recorded Delay': cur_delay,
        'Predicted Delay (HOURS)': predicted_delay,
        'Total Delay': cur_delay + predicted_delay,
        'Date Predicted': TODAY
        }
        predictions_df = pd.concat([predictions_df,pd.DataFrame([predict_row])],ignore_index=True)

    raw_df = predictions_df.copy(deep=True)

    print("Raw features generated!")
    with pd.ExcelWriter("GeminiTracker Analysis.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        # Write the DataFrame to a new or existing sheet
        raw_df.to_excel(writer, sheet_name=f'Raw_Data_{TODAY.strftime('%d%m%Y')}', index=False)

    print(f"DataFrame written to Raw_Data_{TODAY.strftime('%d%m%Y')}' sheet in the existing workbook.")

## 7. Use Random Forest Model to predict delay category
print("Making predictions...")

## Extract Training data set.
df = pd.read_excel("GeminiTracker Analysis.xlsx", sheet_name="Training Data", header = 0)

# convert dates using pd.to_datetime()
df["Prev ETB"] =pd.to_datetime(df["Prev ETB"])
df["CST ETB"] = pd.to_datetime(df["CST ETB"])
df["Date Predicted"] = pd.to_datetime(df["Date Predicted"])

# Extract the Week, Day and Hour from Prev and CST ETB
df["Prev Week"] = df["Prev ETB"].dt.isocalendar().week
df["Prev Day"] = df["Prev ETB"].dt.dayofweek
df["Prev Hour"] = df["Prev ETB"].dt.hour

df["CST Week"] = df["CST ETB"].dt.isocalendar().week
df["CST Day"] = df["CST ETB"].dt.dayofweek
df["CST Hour"] = df["CST ETB"].dt.hour

# Obtain Prev Delay by subtracting Actual Prev Delay from Fcst Prev Delay
df["Prev Delay"] = df["Actual Prev Delay"] - df["Fcst Prev Delay"]

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['SSY','Prev. Port', 'Port', 'LEG'], drop_first=True)
print(df.columns)

#Define Features (X) and Target (y)
X = df.drop(columns=["Prev ETB", "VSL NAME", "VOY NO", "CST ETB", "Fcst Prev Delay", "Actual Prev Delay","Weather Multiplier","Total Time Needed",
                     "Predicted Delay (HOURS)","Total Delay","Date Predicted","Actual Delay","Actual Time Needed",
                     "Difference","Total Delay Category","Actual Delay Category","Correct?","Comments","Actual ETD","Latest ETD (16 knots)","Predicted Delay Category"],errors='ignore')

y = df["Actual Delay Category"]

key = predictions_df[["SSY", "VOY NO", "VSL NAME", "LEG"]]

# Extract the Week, Day and Hour from Prev and CST ETB
predictions_df["Prev Week"] = predictions_df["Prev ETB"].dt.isocalendar().week
predictions_df["Prev Day"] = predictions_df["Prev ETB"].dt.dayofweek
predictions_df["Prev Hour"] = predictions_df["Prev ETB"].dt.hour

predictions_df["CST Week"] = predictions_df["CST ETB"].dt.isocalendar().week
predictions_df["CST Day"] = predictions_df["CST ETB"].dt.dayofweek
predictions_df["CST Hour"] = predictions_df["CST ETB"].dt.hour

# Obtain Prev Delay by subtracting Actual Prev Delay from Fcst Prev Delay
predictions_df["Prev Delay"] = predictions_df["Actual Prev Delay"] - predictions_df["Fcst Prev Delay"]
# One-hot encode categorical variables
predictions_df = pd.get_dummies(predictions_df)

predictions_df = predictions_df.reindex(columns=X.columns, fill_value=0)

# Load the model
loaded_model = joblib.load('vessel_delay_category_RFM_13032025.pkl')

# Load the label encoder
loaded_label_encoder = joblib.load('label_encoder_13032025.pkl')

# Obtain model feature importance values
def plot_feature_importance(model, names, threshold = None):
    feature_importance_df = pd.DataFrame.from_dict({'feature_importance': model.feature_importances_,
                                                    'feature': names}).set_index('feature').sort_values('feature_importance', ascending = False)

    if threshold is not None:
        feature_importance_df = feature_importance_df[feature_importance_df.feature_importance > threshold]

    fig = px.bar(
        feature_importance_df,
        text_auto = '.2f',
        labels = {'value': 'feature importance'},
        title = 'Feature importances'
    )

    fig.update_layout(showlegend = False)
    fig.show()

plot_feature_importance(loaded_model, predictions_df.columns)

# Make predictions
predicted_category = loaded_model.predict(predictions_df)

# Decode the predicted category
predicted_category_label = loaded_label_encoder.inverse_transform(predicted_category)

predictions_df["Predicted Delay Category"] = predicted_category_label

predictions_df = pd.concat([key, predictions_df],axis=1)

print("All predictions done!")
print(predictions_df.head())

with pd.ExcelWriter("GeminiTracker Analysis.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    # Write the DataFrame to a new or existing sheet
    predictions_df.to_excel(writer, sheet_name=f'Predicted_Delays_{TODAY.strftime('%d%m%Y')}', index=False)

print(f"DataFrame written to Predicted_Delays_{TODAY.strftime('%d%m%Y')}' sheet in the existing workbook.")

