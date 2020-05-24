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
      print(ticker)    
      if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
        try:
          df = web.DataReader(ticker.replace('.','-'), 'yahoo', start, end)
          df.to_csv(f'stock_dfs/{ticker}.csv')
        except:
          print(f'Problems found when retrieving data for{ticker}. Skipping!')
        else:
          print('Already have {}'.format(ticker))


#get_data_from_yahoo()
def compile_data():
  with open("sp500tickers.pickle", "rb") as f:
    tickers = pickle.load(f)

# Begin dataframe
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

# Print counter, intervals of 10
#    if count % 10 == 0:
#      print(count)

    print(mainDataSet.head())
  mainDataSet.to_csv('sp500_joined_closes.csv')
compile_data()  

	
