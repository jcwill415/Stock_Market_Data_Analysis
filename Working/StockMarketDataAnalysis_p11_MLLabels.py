import bs4 as bs
import pickle
import requests
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import collections
from collections import Counter

def save_sp500_tickers():
  resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
  soup = bs.BeautifulSoup(resp.text, "lxml")
  table = soup.find('table', {'id': 'constituents'})
  tickers = []
  for row in table.findAll('tr')[1:]:
    ticker = row.find_all('td')[0].text.strip()
    tickers.append(ticker)

  with open("sp500tickers.pickle", "wb") as f:
    pickle.dump(tickers, f)

  print(tickers)

  return tickers

#save_sp500_tickers()
  

# Get data from Yahoo and call SP500 tickers list as sp500

def get_data_from_yahoo(reload_sp500 = False):
  if reload_sp500:
    tickers = save_sp500_tickers()
  else:
    with open("sp500tickers.pickle", "rb") as f:
      tickers = pickle.load(f)

  
# Take all of the data for stocks and store in a directory
# Working with API, parsing website, take entire dataset and store locally
# Here we will look at Adjusted Close, but we can look at other columns later

  if not os.path.exists('stock_dfs'):
    os.makedirs('stock_dfs')

  start = dt.datetime(2000,1,1)
  end = dt.datetime(2020,5,23)

# Grab all ticker data

  for ticker in tickers:
#    try:
      print(ticker)
#    except KeyError:
#      pass
    
      if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
        try:
          df = web.DataReader(ticker.replace('.','-'), 'yahoo', start, end)
          df.to_csv(f'stock_dfs/{ticker}.csv')
        except:
          print(f'Problems found when retrieving data for{ticker}. Skipping!')
#          df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
          print('Already have {}'.format(ticker))


#get_data_from_yahoo()

def compile_data():
  with open("sp500tickers.pickle", "rb") as f:
    tickers = pickle.load(f)

# Begin dataframe

#  main_df = pd.DataFrame()
  mainDataSet = pd.DataFrame()

# Count in SP500 tickers list
  for count, ticker in enumerate(tickers):
    fileDataSet = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
    fileDataSet.set_index('Date', inplace = True)

    fileDataSet.rename(columns = {'Adj Close':ticker}, inplace = True)
    fileDataSet.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace = True)

    if mainDataSet.empty:
      mainDataSet = fileDataSet
    else:
      mainDataSet = mainDataSet.join(fileDataSet)
    print(mainDataSet.head())
  mainDataSet.to_csv('sp500_joined_closes.csv')
##compile_data()  
#    try:
#     df = pd.read_csv('stock_dfs/{}.csv'.format(ticker.replace('.','-')))
#    df.set_index('Date', inplace = True)

#      df.rename(columns = {'Adj Close': ticker}, inplace = True)
#      df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace = True)
# Joining dataframes together
#      if main_df.empty:
#        main_df = df
#      else:
#        main_df = main_df.join(df, how = 'outer')
#        main_df.join(df, how = 'outer')
#    except:
#        print('stock_dfs/{}.csv'.format(ticker) + 'not found')
        
# Print intervals of 10
#    if count % 10 == 0:
#      print(count)

#  print(main_df.head())
#  main_df.to_csv('sp500_joined_closes.csv')
#compile_data()

      





# Machine Learnings: Preprocess Data for Trading Strategy

# Disclaimer: This script is for educational purposes only.
# To build more accurate ML models, you might focus on specific time frames (i.e. 1-2 yrs)
# Further, you might need more data.
# We use one-day data because it is available for free.

# Create features/labels, use ML for trading strategy & possible investments.	
# Theory: groups of companies might move up/down together (using pricing data).
# Take S&P500 dataset (closing price over time, 2000-2020), use machine learning.
# Normalize dataset by converting pricing data to % change
# Features = define, labels = target
# Labels = Buy, Sell, or Hold
# Take feature data and determine label by asking:
# "Within next 7 trading days did price go up more than x% (i.e. 2%)?"
# If yes, sell company.
# If no, hold company.

import numpy as np
import pandas as pd
import pickle

# Each model generated per company
# Each company model considers pricing data from entire SP500 dataset
# To look further into the future, i.e. 30 days, change to "hm_days = 30:
def process_data_for_labels(ticker):
    hm_days = 7

    fileDataSet = pd.read_csv('sp500_joined_closes.csv', index_col = 0)
    tickers = fileDataSet.columns.values.tolist()
    fileDataSet.fillna(0, inplace = True)
##    df = pd.read_csv('sp500_joined_closes.csv', index_col = 0)
##    tickers = df.columns.values.tolist()
##    df.fillna(0, inplace = True)

# The range will go up to a certain point (for 7 days)  
# Create custom dataset to predict future values based on percentage change
# Value in percent change = price in two days from now
# less today's price, divided by today's price, multiplied by 100.

    for i in range(1, hm_days+1):
        fileDataSet['{}d'.format(ticker, i)] = (fileDataSet[ticker].shift(-i) - fileDataSet[ticker]) / fileDataSet[ticker]

    fileDataSet.fillna(0, inplace = True)
    return tickers, fileDataSet
##    for i in range(1, hm_days+1):
##        df['{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
##
##    df.fillna(0, inplace = True)
##    return tickers, df

# Here we look at Apple (ticker = AAPL)
process_data_for_labels('MMM')

# Next, generate labels for targets using machine learning for investing with Python
# Based on percent change pricing information, should we buy, sell, or hold company?
# See mapping functions tutorials at https://pythonprogramming.net/python-function-mapping-pandas/
# Args and kwargs tutorials at https://pythonprogramming.net/args-kwargs-intermediate-python-tutorial/
# We will also back-test out strategy.
# Example here using percent change = 2% in a week.
# If company stock price changes by 2% in 7 days get out of position for that company.
# 0 = hold, -1 = sell, +1 = buy

def buy_sell_hold(*args):

    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

# Map the function above to a column
def extract_featuresets(ticker):
    tickers, fileDataSet = process_data_for_labels(ticker)
##    tickers, df = process_data_for_labels(ticker)

# Define new column, value = mapped function using arg
# Arg = 7-day & change for future price; Pass paramters to function
# Generate labels for buy, sell, or hold
# Update this to clean up list iteration below
##    fileDataSet['{}_target.format(ticker)'] = list(map( buy_sell_hold,
##                                                      fileDataSet['{}_1d'.format(ticker)],
##                                                      fileDataSet['{}_2d'.format(ticker)],
##                                                      fileDataSet['{}_3d'.format(ticker)],
##                                                      fileDataSet['{}_5d'.format(ticker)],
##                                                      fileDataSet['{}_6d'.format(ticker)],
##                                                      fileDataSet['{}_7d'.format(ticker)],
##                                                      ))
##    df['{}_target.format(ticker)'] = list(map( buy_sell_hold,
##                                                      df['{}_1d'.format(ticker)],
##                                                      df['{}_2d'.format(ticker)],
##                                                      df['{}_3d'.format(ticker)],
##                                                      df['{}_5d'.format(ticker)],
##                                                      df['{}_6d'.format(ticker)],
##                                                      df['{}_7d'.format(ticker)],
##                                                      ))
## Loop list iteration?
##    fileDataSet['{}_target'.format(ticker)] = list(map(buy_sell_hold, *[fileDataSet['{}_{}d'.format(ticker, i)] for i in range(1, timeframe+1)])) 
## OR
##    fileDataSet['{}_target'.format(ticker)] = list(map(buy_sell_hold, fileDataSet[['{}.{}d'.format(ticker, i) for i in range(1, hm_days+1)].values))
## OR
    fileDataSet['{}_target'.format(ticker)] = list(map(buy_sell_hold, fileDataSet[[c for c in fileDataSet.columns if c not in tickers]].values))

    vals = fileDataSet['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))
##    vals = df['{}_target'.format(ticker)].values.tolist()
##    str_vals = [str(i) for i in vals]
##    print('Data spread:', Counter(str_vals))

# List of options
    fileDataSet.fillna(0, inplace = True)
    fileDataSet = fileDataSet.replace([np.inf, -np.inf], np.nan)
    fileDataSet.dropna(inplace = True)
##    df.fillna(0, inplace = True)
##    df = df.replace([np.inf, -np.inf], np.nan)
##    df.dropna(inplace = True)

# Normalizing data set for ticker (prices) by using percent change.
##    fileDataSet_vals = fileDataSet[[ticker for ticker in tickers]].pct_change()
    fileDataSet_vals = fileDataSet[[ticker_name for ticker_name in tickers]].pct_change()

    fileDataSet_vals = fileDataSet_vals.replace([np.inf, -np.inf], 0)
    fileDataSet_vals.fillna(0, inplace = True)

    X = fileDataSet_vals.values
    y = fileDataSet['{}_target'.format(ticker)].values

    return X, y, fileDataSet

extract_featuresets('MMM')
##    df_vals = df[[ticker for ticker in tickers]].pct_change()
##    df_vals = df_vals.replace([np.inf, -np.inf], 0)
##    df_vals.fillna(0, inplace = True)
##
##    X = df_vals.values
##    y = df['{}_target'.format(ticker)].values
##
##    return X, y, df
##
##extract_featuresets('AAPL')
