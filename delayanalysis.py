## 1. Create Gemini Dataframe using provided input date range
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

gemini_df = pd.read_excel("GeminiTracker Analysis.xlsx",sheet_name="Dashboard",header=0)

START = "17-02-2025"
END = "21-02-2025"
# replace all OMIT with 0
gemini_df.replace('OMIT', 0, regex = True, inplace = True)
gemini_df['Date'] = pd.to_datetime(gemini_df['Date'])

start_date = pd.to_datetime(START)
end_date = pd.to_datetime(END)

gemini_df = gemini_df[(gemini_df["Date"] >= start_date) & (gemini_df["Date"] <= end_date)]

## 2. Create Port Dataframe by pivoting port-delay columns into long format
pivoted_data = []
pairs = [('leg1','d1'),('leg2','d2'),('leg3','d3'),('leg4','d4')]

for leg_col, delay_col in pairs:
    df_pivot = gemini_df[['VSL NAME', 'SSY', 'OPR','Date', leg_col, delay_col]].copy()
    df_pivot = df_pivot.rename(columns={leg_col:'Port',delay_col:'Delay'})
    df_pivot["Leg"] = leg_col.strip('leg')
    
    pivoted_data.append(df_pivot)
 
port_data = pd.concat(pivoted_data, ignore_index = True)
port_data = port_data[port_data['Delay'].notnull()]

## 3. Create Operator Delay Heatmap
OPR_group = port_data.groupby(['OPR', 'Date'], as_index=False).agg({'Delay': 'mean'})
OPR_group['Delay'] = OPR_group['Delay'].round(2)
OPR_group['Date'] = OPR_group['Date'].dt.date

port_group = port_data.groupby(['Port', 'Date'], as_index=False).agg({'Delay': 'mean'})
port_group['Delay'] = port_group['Delay'].round(2)
port_group['Date'] = port_group['Date'].dt.date

df_pivot = OPR_group.pivot(index='OPR', columns =['Date'], values = 'Delay')

sns.heatmap(df_pivot, annot=True, cmap='YlGnBu', fmt = 'g', cbar_kws={'label': 'Mean Delay (hours)'})
plt.title('Operator Delays Heatmap')
plt.xlabel('Date')
plt.ylabel('Operator')

plt.show()

## 4. Visualise Mean Delays for Ports in China
import matplotlib.dates as mdates

ts_df = port_group[port_group["Port"].str.contains("CN")][["Date","Delay","Port"]]

unstacked = ts_df.groupby(["Date","Port"])["Delay"].mean().unstack()
unstacked.index = pd.to_datetime(unstacked.index)
#resample for business days, skipping weekends. Can change to resample monthly.
unstacked.resample('B', label='right', closed='right').mean().plot(cmap='tab10', alpha = 0.6, linestyle = "--", marker = 'o')
    
plt.title("Mean Delay for Ports")
plt.xlabel("Date")
plt.ylabel("Mean Delay (in hours)")
plt.legend(loc="upper left")

plt.xticks(rotation=45)
plt.style.use("ggplot")

## 5. Visualise Mean Delays for Ports in NE4 Shipsystem
ssy = "NE4"
box_df = port_data[port_data['SSY'].str.contains(ssy)]

unstacked = box_df.groupby(["Date","SSY","VSL NAME"])["Delay"].mean().reset_index()
unstacked.index = pd.to_datetime(unstacked.index)
#resample for business days, skipping weekends. Can change to resample monthly.
unstacked.resample('B', label='right', closed='right').last()

#Plot the graph
plt.figure(figsize=(14, 7))
sns.lineplot(x='Date', y='Delay', hue='VSL NAME', data=unstacked, marker='o')
plt.title(f'Mean Delay per {ssy} Vessel Over Time')
plt.ylabel('Mean Delay')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.legend(title='Vessel', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

## 6. Create Boxplot of Delays for all Legs of the NE Group Shipsystems, omitting values of 0.

group_ssy = ssy[:-1]
box_df = port_data[port_data['SSY'].str.contains(group_ssy) & port_data['Delay'] != 0]
sns.boxplot(data = box_df, hue = 'SSY', x = 'Leg', y = 'Delay')
plt.title("Boxplot of Delays (Grouped by SSY)")
plt.ylabel("Delay (hours)")
plt.xlabel("Delay at Leg")

plt.legend(title = 'SSY', bbox_to_anchor =(1,1), loc = 'upper right')
plt.show()

## 7. View Total Leg 1 Delays by Shipsystem.
grouped_ssy = gemini_df.groupby('SSY')['d1'].sum()
ax = grouped_ssy.plot(kind= "bar", color = 'skyblue')

# Get the bars
rects = ax.patches

# Add value labels on top of bars
for bar in rects:
    height = bar.get_height()
    ax.annotate(
        f'{height:.1f}',  # Format to 1 decimal
        (bar.get_x() + bar.get_width() / 2, height),
        ha='center', va='bottom'
    )
   
    
plt.title('Total Leg1 Delays by Shipsystem')
plt.xlabel('Shipsystem')
plt.ylabel('Total Delay Duration (hours)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout() 
plt.show()

