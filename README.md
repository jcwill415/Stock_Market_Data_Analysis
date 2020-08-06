# STOCK MARKET DATA ANALYSIS
Use Python to scrape data and join with financial data from Yahoo Finance (or another finance) API. Use data manipulation and visualization for financial and investment analysis (i.e. compare rates of return, calculate risk, build trading algorithms, and make investment decisions).  

## How to Run
* Use the `Stock_Market_Data_Analysis.ipynb` file to run the program in Jupyter Notebook. Use the ".py" file (`Stock_Market_DataAnalysis_DataVisualization.py`) to run the program only in Python.
* For a walk through of the project - how to install Python, the necessary packages, Jupyter, and run the notebook file - please see my **[Demo Screen Recording here](https://www.screencast.com/users/jcwill415/folders/Default/media/678a3aad-e829-4f00-ab63-2e53ed84b5d3/)**.

  * For users running the project with Jupyter:

1) After downloading the files, if Python is not installed, please install it by **[downloading Python here](https://www.python.org)**.
2) After installing Python, open a Python shell and run the following commands to install the necessary packages to run my program:
* &nbsp;&nbsp;&nbsp;&nbsp;`pip install pandas`
* &nbsp;&nbsp;&nbsp;&nbsp;`pip install pandas-datareader`
* &nbsp;&nbsp;&nbsp;&nbsp;`pip install beautifulsoup4`
* &nbsp;&nbsp;&nbsp;&nbsp;`pip install scikit-learn`
* &nbsp;&nbsp;&nbsp;&nbsp;`pip install numpy`
* &nbsp;&nbsp;&nbsp;&nbsp;`pip install matplotlib`
* &nbsp;&nbsp;&nbsp;&nbsp;`pip install mplfinance`
* &nbsp;&nbsp;&nbsp;&nbsp;`pip install mpl-finance`
* &nbsp;&nbsp;&nbsp;&nbsp;`pip install yfinance`
* &nbsp;&nbsp;&nbsp;&nbsp;`pip install jupyter`
3) Once everything is installed, change directory (cd) to navigate to where the project has been downloaded. 
4) Locate the ".ipynb" file `Stock_Market_Data_Analysis.ipynb` and run Jupyter with the command: `jupyter notebook` in your terminal or cmd; this will take you to the project in Jupyter Notebook, opening up a browser.

5) Once Jupyter opens in the browser, you should see the `Stock_Market_Data_Analysis.ipynb` notebook file. Double click to open the file.
6) To run the program, select "Cell," and "Run All."
7) For additional charts, please run the `Stock_Market_Data_Analysis_DataVisualization.ipynb` file in Jupyter Notebook. Here I focus mainly on data visualizations for large cap tech stocks (i.e. Apple, Google, Facebook), and model various chart types. I also begin the next part of the project: Fundamental Analysis. In this part, we will be using finanical statements from the [SEC Edgar](https://www.sec.gov/edgar/searchedgar/companysearch.html) website.

## Project Summary 
* First, this project will focus on technical analysis, measuring stock price data for movement (volatility) and volume.
* Second, I plan to expand this project in the future, to include a branch with fundamental analysis, to look more in depth at financial statement analysis.
* Third, I am interested in expanding the analysis to include Python for cryptocurrencies, such as financial and investment analysis for ICOs, and predicting crypto prices. 

##### <b>DISCLAIMER:</b> I am not a financial adviser nor a CPA. This project is for educational purposes only. Investing of any kind involves risk. Although it is possible to minimize risk, your investments are solely your responsibility. It is very important to conduct your own research. I am merely sharing my project/opinion with no guarantee of either gains or losses on any investments.

## Technical Summary
* Back end language: Python (the version used here is Python 3.7.6)
* Dataset: csv, stock price data via Yahoo Finance
* Packages: Pandas/NumPy; Scikit-learn for Machine Learning in Python; Matplotlib (and mplfinance) for data manipulation and visualization.
* (For Code Louisville: 5+ commits using Git.)

## Features
The script will scrape data for S&P500 tickers, pull financial data from Yahoo Finance API, and download into a csv file. It will also manipulate/clean data, and merge multiple data frames into one large csv file. The script uses for loops, dictionaries, and error handling. Further, there is additional data visualization in the "Stock_Market_Data_Analysis_DataVisualization.ipynb" Jupyter Notebook file; this is done using matplotlib to build various stock charts (i.e. line charts, bar charts, moving average bar charts, candlestick charts). Additional features are highlighted below:
* Scrape stock tickers from web (i.e. Wikipedia). 
* For this project, the SP500 list is from: **[List of S&P 500 Companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)**.
* Use Requests to scrape data for Beautiful Soup to parse.
* Using this soup object, navigate/search HTML for data you want to pull. 
* Create directory with stock price close data for all 500 companies over time (i.e. 01/01/2000 to 05/24/2020). 
* Build quantitative models to conduct financial and investment analysis such as risk and rates of return, and build a basic trading strategy. 

## Milestones
* Import needed packages/modules
<b>Main Requirements for this Program:</b>
   * `pandas`
   * `pandas-datareader`
   * `beautifulsoup4`
   * `scikit-learn`
   * `numpy`
   * `yfinance` (or another finance API)
   * `jupyter` 
   * `matplotlib`
   * `mplfinance`, `mpl_finance`
- mplfinance requires [matplotlib](https://pypi.org/project/matplotlib/) and [pandas](https://pypi.org/project/pandas/)
- Installation: `pip install --upgrade mplfinance`

* You will see the first cell in the Jupyter notebook file will import all the necessary packages to run the program:
```python
import requests
import datetime as dt
import os
import io
import pandas as pd
import pandas_datareader.data as web
import pandas.plotting
from pandas.plotting import register_matplotlib_converters
import pandas.testing #pandas.testing.assert_frame_equal
from pandas.testing import assert_frame_equal #assert_frame_equal

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib import style

import mplfinance as mpf
import mpl_finance as mplf
from mpl_finance import candlestick_ohlc

import collections
from collections import Counter

import sklearn
from sklearn import svm, neighbors
from sklearn.svm import LinearSVC 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

style.use('ggplot')
```
* Scrape data using `BeautifulSoup`
* Download financial data from API
* Join tickers with stock price data
* Visualize data using `Matplotlib`
* Create basic trading strategy
## Requirements
* Build webscraper to get data for S&P500 List.
* Use Yahoo Finance (or other) API `yfinance` for financial data.
* Use `Pandas` to join stock tickers with financial data.
* Analyze data with various quantitative models to calculate risk and make investment decisions.
* Download data as csv and read. 
  * Build quantitative models to predict returns & evaluate risk. 
  * Run basic Value at Risk (VaR) calculations, Monte Carlo Simulations.
  * Looking at news sentiment as a proxy for market volatility.
  * Graph/visualize data.

<b> NOTE:</b> If you are new to Python, check out the **[Python Programming Fundamentals website](https://pythonprogramming.net/introduction-learn-python-3-tutorials/using)** for tutorials. You will need to review up to installing Python packages and modules with pip. 

## <b>Instructions (Overview):</b>
* <b>Step 1:</b> Intro to Using Python for Finance
* <b>Step 2:</b> Handling and Graphing Data
* <b>Step 3:</b> Stock Data Manipulation
* <b>Step 4:</b> Data Resampling
* <b>Step 5:</b> S&P500 List Automation
* <b>Step 6:</b> Getting S&P500 Stock Price Data
* <b>Step 7:</b> Combine DataFrames for S&P500 List and Stock Price Data
### Python for Machine Learning
* <b>Step 8:</b> Building S&P500 Correlation Table
* <b>Step 9:</b> ML: Preprocess Stock Market Data
* <b>Step 10:</b> ML: Create Target Function 
* <b>Step 11:</b> ML: Create Labels
* <b>Step 12:</b> ML Algorithm: Mapping Relationships for Stock Prices

## The New API (mplfinance)
There is a new **[matplotlib finance (mplfinance) API](https://github.com/matplotlib/mplfinance)** that has made creating financial plots easier.
Some of the updates include automatic features for the user, and improved interacing with Pandas DataFrames.
* Note: This repository is using both modules (the old and new matplotlib/mplfinance APIs), because I am working to update the candlestick charts in the script.
* For more information on the old API, please see below.

According to the **[matplotlib/mplfinance repo](https://github.com/matplotlib/mplfinance)**, the conventional way to import the new API is as follows:
```python
    import mplfinance as mpf
```
The most common usage is then to call
```python
    mpf.plot(data)
```
where `data` is the `Pandas DataFrame` object, which contains Open, High, Low and Close pricing data, using a Pandas `DatetimeIndex`.

Further details on how to call the new API can be found below under Basic Usage on the [matplotlib/mplfinance](https://github.com/matplotlib/mplfinance) repo, as well as in the jupyter notebooks in the examples folder.

## <b> Resources / Tutorials:</b>

365 Careers (2020). **[Python for Finance: Investment Fundamentals & Data Analytics](https://www.udemy.com/course/python-for-finance-investment-fundamentals-data-analytics/)**

B., V. (2019). **[Stock Market Data and Analysis in Python](https://www.datacamp.com/community/news/stock-market-data-and-analysis-in-python-4hzkx0wemva)**

**[Beautiful Soup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)**

Boller, K. (2018). **[Python for Finance: Stock Portfolio Analyses](https://towardsdatascience.com/python-for-finance-stock-portfolio-analyses-6da4c3e61054)**

Danielsson, J (2011). **[Financial Risk Forecasting](https://www.financialriskforecasting.com/code/FRF_in_Python_jupyter.html)**

Dhiman, A. (2019). **[Stock Market Data Analysis with Python](https://www.kaggle.com/dhimananubhav/stock-market-data-analysis-with-python)**

Efstathopoulos, G. (2019). **[Python for Finance, Part I: Yahoo & Google Finance API, Pandas, and Matplotlib](https://learndatasci.com/tutorials/python-finance-part-yahoo-finance-api-pandas-matplotlib/#)**

Huang, S. (2019). **[Best 5 Free Stock Market APIs in 2020](https://towardsdatascience.com/best-5-free-stock-market-apis-in-2019-ad91dddec984)**

Lewinson, E. (2020). **[A Comprehensive Guide to Downloading Stock Prices in Python.](https://towardsdatascience.com/a-comprehensive-guide-to-downloading-stock-prices-in-python-2cd93ff821d4)**

IEX Cloud (2020). **[IEX CLoud API](https://iexcloud.io/)**

Kharkar, R. (2020). **[How to Get Stock Data Using Python](https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75)**

Matplotlib (2020). **[Matplotlib Documentation](https://matplotlib.org/contents.html)**

Miller, C. (2018). **[Introduction to Stock Market Data Analysis with Python](https://ntguardian.wordpress.com/2018/07/17/stock-data-analysis-python-v2/)**

Miller, C. (2019). **[Pakt Publishing: Training Your Systems with Python Statistical Modeling](https://www.packtpub.com/big-data-and-business-intelligence/training-your-systems-python-statistical-modeling-video)**

O'Keefe, C. (2020). **[Practical Introduction to Web Scraping in Python](https://realpython.com/python-web-scraping-practical-introduction/)**

**[Open Machine Learning Course](https://mlcourse.ai/)**

Pandas (2020). **[Pandas Documentation](https://pandas.pydata.org/docs/)**

Python Programming (2020). **[Pythonprogramming.net](https://pythonprogramming.net)**

> **[Python Programming for Finance](https://pythonprogramming.net/getting-stock-prices-python-programming-for-finance/)**

> **[Python Fundamentals > Basics](https://pythonprogramming.net/introduction-learn-python-3-tutorials/)**
Complete through how to install python packages and programs using pip
(about 10-11 tutorials)

> **[Data Analysis with Python 3 and Pandas](https://pythonprogramming.net/introduction-python3-pandas-data-analysis/)**

> **[Data Visualization with Python and Matplotlib](https://pythonprogramming.net/matplotlib-intro-tutorial/)**

Quandl (2020). **[Quandl API: Core Financial Data](https://www.quandl.com/)**

Quandl (2020). **[Get Financial Data Directly into Python.](https://www.quandl.com/tools/python)**

Vaidyanathan, V. (2020). **[Coursera: Investment Management with Python and Machine Learning Specialization](https://www.coursera.org/specializations/investment-management-python-machine-learning)**

Vaidyanathan, V. (2020b). **[Coursera: Course 1 - Introduction to Portfolio Construction and Analysis with Python](https://www.coursera.org/learn/introduction-portfolio-construction-python)**

Vaidyanathan, V. (2020c). **[Coursera: Course 2 - Advanced Portfolio Construction and Analysis with Python](https://www.coursera.org/learn/advanced-portfolio-construction-python)**

Vaidyanathan, V. (2020d). **[Coursera: Course 3 - Python and Machine Learning for Asset Management](https://www.coursera.org/learn/python-machine-learning-for-investment-management)**

Vaidyanathan, V. (2020e). **[Coursera: Course 4 - PYthon and Machine Learning for Asset Management with Alternative Data Sets](https://www.coursera.org/learn/machine-learning-asset-management-alternative-data)**
