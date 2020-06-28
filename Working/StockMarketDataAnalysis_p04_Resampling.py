import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

import mplfinance
from mpl_finance import candlestick_ohlc

import pandas as pd
import pandas_datareader.data as web
import pandas.testing

style.use('ggplot')

df = pd.read_csv('C:/Users/JCW/Desktop/Stock_Market_Data_Analysis/tsla.csv', parse_dates = True, index_col = 0)
#df['100ma'] = df['Adj Close'].rolling(window = 100, min_periods = 0).mean()
#print(df.tail(10))

#df_ohlc = df['Adj Close'].resample('10D').mean()

# Resample data for 10 day period
df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace = True)


# Convert datetime object to mdate
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
print(df_ohlc.head())


ax1 = plt.subplot2grid((6,1), (0,0), rowspan = 5, colspan = 1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan = 1, colspan = 1, sharex = ax1)
ax1.xaxis_date()

candlestick_ohlc(ax1, df_ohlc.values, width = 3, colorup = 'g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
                 
#ax1.plot(df.index, df['Adj Close'])
#ax1.plot(df.index, df['100ma'])
#ax2.bar(df.index, df['Volume'])

plt.show()


