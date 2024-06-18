import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings


# Download data from Yahoo Finance
data = yf.download("AAPL", start="2014-01-01", end="2024-06-12")
plt.figure(figsize=(16,8))
plt.plot(data['Adj Close'])
plt.title('Apple Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show() 


df = data.reset_index()
df = df[['Date', 'Adj Close']]
df = df.rename(columns={'Date' : 'ds', 'Close': 'y'})

warnings.filterwarnings("ignore")

model = Prophet() 
model.fit(df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
figure1 = model.plot(forecast)

