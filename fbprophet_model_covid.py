
# IMPORT LIBRARIES
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from prophet import Prophet


# LOAD DATA

df = pd.read_csv('covid_19_clean_complete.csv')


# PREPROCESSING
df['Date'] = pd.to_datetime(df['Date'])

total = df.groupby(['Date'])[['Confirmed','Deaths','Recovered','Active']].sum().reset_index()

# Prophet format
df_prophet = total[['Date','Confirmed']].rename(columns={'Date':'ds','Confirmed':'y'})


# MODEL TRAINING
m = Prophet()
model = m.fit(df_prophet)

# FUTURE DATAFRAME
future = model.make_future_dataframe(periods=100, freq='D')


# PREDICTION
forecast = model.predict(future)

print(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail())

# PLOTS
model.plot(forecast)
plt.title("COVID Forecast")
plt.show()

model.plot_components(forecast)
plt.show()

# Changepoints
from prophet.plot import add_changepoints_to_plot

fig = model.plot(forecast)
add_changepoints_to_plot(fig.gca(), model, forecast)
plt.show()

# CROSS VALIDATION
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

df_cv = cross_validation(
    model,
    horizon='30 days',
    period='15 days',
    initial='90 days'
)

df_performance = performance_metrics(df_cv)
print(df_performance.head())

plot_cross_validation_metric(df_cv, metric='mape')
plt.show()