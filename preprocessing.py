
import pandas as pd
import numpy as np
import yfinance as yf




data = yf.download(
    tickers=['BTC-USD'],
    # use "period" instead of start/end
    # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    # (optional, default is '1mo')
    period="1mo",
    # fetch data by interval (including intraday if period < 60 days)
    # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    # (optional, default is '1d')
    interval="1h")

header = ["day " + str(i) for i in range(1, 32)]

closes = data['Close'].values
df = []
for i in range(30, len(closes)):
    arr = []
    for j in range(i-30, i+1):
        arr.append(closes[j])
    df.append(np.array(arr))
df = pd.DataFrame(df, columns=header)

data_x = []
data_y = []
def saveToLSTMData(x):
    datapoint = []
    for i in range(1, 31):
        timestamp = []
        timestamp.append(x['day ' + str(i)])
        datapoint.append(timestamp)
    data_x.append(datapoint)
    data_y.append(x['day 31'])

df.apply(lambda x : saveToLSTMData(x), axis=1)
data_x = np.array(data_x)
data_y = np.array(data_y)

print()









