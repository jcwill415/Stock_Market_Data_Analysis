import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style

import mplfinance
from mpl_finance import candlestick_ohlc

#import plotly

import pandas as pd
import pandas_datareader
from pandas.testing import assert_frame_equal
import pandas_datareader.data as web
style.use('ggplot')

df = pd.read_csv('tsla.csv', parse_dates = True, index_col = 'Date')
start = dt.datetime(2000,1,1)
end = dt.datetime(2020,5,22)
df = web.DataReader('TSLA', 'yahoo', start, end)

print(df.tail(10))

df = web.DataReader('TSLA', 'yahoo', start, end)
df.to_csv('C:/Users/JCW/Desktop/tsla.csv')
df = pd.read_csv('C:/Users/JCW/Desktop/tsla.csv', parse_dates = True, index_col = 'Date')

df['100ma'] = df['Adj Close'].rolling(window = 100, min_periods = 0).mean()
df.dropna(inplace = True)
print(df.tail())

ax1 = plt.subplot2grid((6,1), (0,0), rowspan = 5, colspan = 1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan = 1, colspan = 1, sharex = ax1)

ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])
plt.show()



df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = pd['Volume'].resample('10D').sum()
print(df_ohlc.head())


    
