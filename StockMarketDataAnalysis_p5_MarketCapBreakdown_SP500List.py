# Visualizating Stock Market Data Using S&P 500 and Quandl Financial Data
# MarketCap Breakdown of SP500

import numpy as np
import pandas as pd
import bs4 as bs
import quandl
quandl.ApiConfig.api_key = "3YQ8M8hosvVVj7sKNZPU"

table = pd.read_html('https://en.wikipedia.org')

# Download financial data from Quandl/Sharadar
table = quandl.get_table('SHARADAR/SF1', paginate = True)

# Grab most recent annual data ('MRY is annual data')
stock_df = table[(table['calendardate'] == '2018-12-31 00:00:00') & (table['dimension'] == "MRY")]

# Scrape S&P 500 tickers from Wikipedia
import requests
import re

page_link = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
page_response = requests.get (page_link, timeout = 1000)
page_content = bs (page_response.content, 'lxml')

