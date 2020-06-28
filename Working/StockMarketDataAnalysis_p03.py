import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style

import pandas as pd
import pandas_datareader.data as web
import pandas.testing

style.use('ggplot')

df = pd.read_csv('C:/Users/JCW/Desktop/Stock_Market_Data_Analysis/CompanyData/tsla.csv', parse_dates = True, index_col = 0)
df['100ma'] = df['Adj Close'].rolling(window = 100).mean()
df['100ma'] = df['Adj Close'].rolling(window = 100, min_periods = 0).mean()
print(df.tail(10))

ax1 = plt.subplot2grid((6,1), (0,0), rowspan = 5, colspan = 1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan = 1, colspan = 1, sharex = ax1)

ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])

ax2.bar(df.index, df['Volume'])
plt.show()
