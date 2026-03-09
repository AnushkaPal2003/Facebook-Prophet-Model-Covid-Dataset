

!pip  install pystan

!python -m pip install prophet

import prophet

dir(prophet)

# Commented out IPython magic to ensure Python compatibility.
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

df=pd.read_csv('/content/covid_19_clean_complete.csv')
df.head()

df.info()

"""Date is main , then confirmed deaths recovered active"""

df['Date']=pd.to_datetime(df['Date'])

df.info()

df.isnull().sum()

df.describe()

# To find unique dates number
df['Date'].nunique()

df.head(2)

"""Grouping all the important dependent features with respect to Date."""

total=df.groupby(['Date'])[['Confirmed','Deaths','Recovered','Active']].sum().reset_index()

total

"""Extract data from worldometers and do this by web scraping but right now we are doing by kaggle dataset.

# FACEBOOK PROPHET MODEL

For FBM , we need a time series dataset and dependent variable

here date is TS and Confirmed, detahs, recovered, active are dependent variable

We will do only one depenedent-Confirmed
"""

df_prophet=total.rename(columns={'Date':'ds','Confirmed':'y'})

df_prophet

from prophet import Prophet

m=Prophet()

model=m.fit(df_prophet)

model.seasonalities

# We want for daily basis

len(df_prophet)

future_global=model.make_future_dataframe(periods=100, freq='D') # Periods = 100 days and daily basis

future_global

"""188 days earlier and 288 days now, 100 extra days are added

#Prediction
"""

prediction=model.predict(future_global)

prediction

"""yhat means actual pred and yhat low and upper are the predicted lowest and upper"""

prediction[['ds','yhat','yhat_lower','yhat_upper']].tail()

# plot the model
model.plot(prediction)
plt.show()

"""In November, there were surge in the covid cases, touching atleast 3.5 Lakh per day."""

model.plot_components(prediction)
plt.show()

"""The cases were rising mostly on Saturday and that make sense because, on weekends people are more likely to go outside and that is why total ban was observed by the government"""

from prophet.plot import add_changepoints_to_plot

fig=model.plot(prediction)
a=add_changepoints_to_plot(fig.gca(),model,prediction)
plt.show()

from prophet.diagnostics import cross_validation

df_cv=cross_validation(model,horizon='30 days',period='15 days',initial='90 days')

"""This performs everything internally"""

df_cv

from prophet.diagnostics import performance_metrics

df_performance=performance_metrics(df_cv)

df_performance

from prophet.plot import plot_cross_validation_metric

df_performance=plot_cross_validation_metric(df_cv,metric='mape')

"""MAPE is 0.125 that means very less error so the model is working well."""

