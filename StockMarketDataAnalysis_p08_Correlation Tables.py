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
from matplotlib import style
style.use('ggplot')

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
    print(mainDataSet.head())
  mainDataSet.to_csv('sp500_joined_closes.csv')
#compile_data()  

# Visualizing data from the SP500 close price csv 
def visualize_data():
  df = pd.read_csv('sp500_joined_closes.csv')

# Example of plotting one company close price over time for Apple (ticker = AAPL) 
##  df['AAPL'].plot()
##  plt.show()

# Create correlation table for all data in df for SP500 close price
  df_corr = df.corr()
  print(df_corr.head())

# Visualize inner values of dataframe (numpy array of columns and rows)
# Specify figure and define axes using parameters(111) : one subplot is 1x1 for plot 1)
  data = df_corr.values
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)

# Define heatmap using a range from red to yellow to green
# Red = Negative, Yellow = Neutral, Green = Positive
# Colorbar for legend
  heatmap = ax.pcolor(data, cmap = plt.cm.RdYlGn)
  fig.colorbar(heatmap)

# Build graph
  ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor = False)
  ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor = False)
  ax.invert_yaxis()
  ax.xaxis.tick_top()

  column_labels = df_corr.columns
  row_labels = df_corr.index
# Make x labels vertical, set limit of colors (-1 = min, 1 = max)
  ax.set_xticklabels(column_labels)
  ax.set_yticklabels(row_labels)
  plt.xticks(rotation = 90)
  heatmap.set_clim(-1, 1)
  plt.tight_layout()
  plt.show()
  
visualize_data()

# Next create features/labels, use ML for trading strategy & possible investments.	
