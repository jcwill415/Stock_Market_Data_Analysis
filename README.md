# STOCK MARKET DATA ANALYSIS
Use Python to scrape data and join with financial data from Yahoo Finance (or another finance) API. Use data manipulation and visualization for financial and investment analysis (i.e. compare rates of return, calculate risk, build trading algorithms, and make investment decisions).  

## How to Run
* Use the "Stock_Market_Data_Analysis.ipynb" file to run the program in Jupyter Notebook. Use the .py file to run the program only in Python.
* For a walk through of the project - how to install Python, the necessary packages, Jupyter, and run the notebook file - please see this screen recording:
<!-- copy and paste. Modify height and width if desired. -->
<iframe class="embeddedObject shadow resizable" name="embedded_content" scrolling="no" frameborder="0" type="text/html" 
        style="overflow:hidden;" src="https://www.screencast.com/users/jcwill415/folders/Default/media/678a3aad-e829-4f00-ab63-2e53ed84b5d3/embed" height="618" width="966" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

  * For users running the project with Jupyter:

1) After downloading the files, if Python is not installed, please install Python from https://www.python.org.
2) After installing Python, open a Python shell and run the following command:
   * "pip install pandas, pandas-datareader, beautifulsoup4, scikit-learn, numpy, matplotlib, mplfinance, yfinance"
3) You will also need to install Jupyter Notebook, so from the shell run the command:
   * "pip install jupyter"
4) Once everything is installed, change directory (cd) to navigate to where the project has been downloaded. 
5) Locate the ".ipynb" file and run Jupyter with the command "jupyter notebook"; this will take you to the project in Jupyter Notebook, opening up a browser.
6) To view the project in Jupyter, select "Cell," and "Run All."

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
* For this project, the SP500 list is from: "List of S&P 500 Companies" at https://en.wikipedia.org/wiki/List_of_S%26P_500_companies.
* Use Requests to scrape data for Beautiful Soup to parse.
* Using this soup object, navigate/search HTML for data you want to pull. 
* Create directory with stock price close data for all 500 companies over time (i.e. 01/01/2000 to 05/24/2020). 
* Build quantitative models to conduct financial and investment analysis such as risk and rates of return, and build a basic trading strategy. 

## Milestones
Import needed packages/modules</br>
<b>Required for this project:</b>
* pandas
* pandas-datareader
* beautifulsoup4
* scikit-learn
* numpy
* matplotlib
* mplfinance
* yfinance (or another finance API) 

## Requirements
* Build webscraper to get data for S&P500 List</b>
* Use Yahoo Finance (or other) API for financial data.
* Use Pandas to join stock tickers with financial data.
* Analyze data with various quantitative models to calculate risk and make investment decisions.
* Download data as csv and read. 
  * Build quantitative models to predict returns & evaluate risk. 
  * Run basic Value at Risk (VaR) calculations, Monte Carlo Simulations.
  * Looking at news sentiment as a proxy for market volatility.
  * Graph/visualize data.

<b> NOTE:</b> If you are new to Python, check out the Python Programming Fundamentals website for tutorials at https://pythonprogramming.net/introduction-learn-python-3-tutorials/using. You will need to review up to installing Python packages and modules with pip. 

## <b>INSTRUCTIONS:</b>
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

## <b> RESOURCES:</b>
RESOURCES:

365 Careers (2020). Python for Finance: Investment Fundamentals & Data Analytics
https://www.udemy.com/course/python-for-finance-investment-fundamentals-data-analytics/

B., V. (2019). Stock Market Data and Analysis in Python
https://www.datacamp.com/community/news/stock-market-data-and-analysis-in-python-4hzkx0wemva

Beautiful Soup Documentation
https://www.crummy.com/software/BeautifulSoup/bs4/doc/

Boller, K. (2018). Python for Finance: Stock Portfolio Analyses
https://towardsdatascience.com/python-for-finance-stock-portfolio-analyses-6da4c3e61054

Danielsson, J (2011). Financial Risk Forecasting
https://www.financialriskforecasting.com/code/FRF_in_Python_jupyter.html

Dhiman, A. (2019). Stock Market Data Analysis with Python
https://www.kaggle.com/dhimananubhav/stock-market-data-analysis-with-python

Efstathopoulos, G. (2019). Python for Finance, Part I: Yahoo & Google Finance API, Pandas, and Matplotlib
https://learndatasci.com/tutorials/python-finance-part-yahoo-finance-api-pandas-matplotlib/#

Huang, S. (2019). Best 5 Free Stock Market APIs in 2020
https://towardsdatascience.com/best-5-free-stock-market-apis-in-2019-ad91dddec984

Lewinson, E. (2020). A Comprehensive Guide to Downloading Stock Prices in Python.
https://towardsdatascience.com/a-comprehensive-guide-to-downloading-stock-prices-in-python-2cd93ff821d4

IEX Cloud (2020). IEX CLoud API
https://iexcloud.io/

Kharkar, R. (2020). How to Get Stock Data Using Python
https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75

Matplotlib (2020). Matplotlib Documentation
https://matplotlib.org/contents.html

Miller, C. (2018). Introduction to Stock Market Data Analysis with Python
https://ntguardian.wordpress.com/2018/07/17/stock-data-analysis-python-v2/

Miller, C. (2019). Pakt Publishing: Training Your Systems with Python Statistical Modeling
https://www.packtpub.com/big-data-and-business-intelligence/training-your-systems-python-statistical-modeling-video

O'Keefe, C. (2020). Practical Introduction to Web Scraping in Python
https://realpython.com/python-web-scraping-practical-introduction/

Open Machine Learning Course
https://mlcourse.ai/

Pandas (2020). Pandas Documentation
https://pandas.pydata.org/docs/

Python Programming (2020). Pythonprogramming.net
https://pythonprogramming.net

> Python Programming for Finance: 
https://pythonprogramming.net/getting-stock-prices-python-programming-for-finance/

> Python Fundamentals > Basics: 
https://pythonprogramming.net/introduction-learn-python-3-tutorials/
Complete through how to install python packages and programs using pip
(about 10-11 tutorials)

> Data Analysis with Python 3 and Pandas:
https://pythonprogramming.net/introduction-python3-pandas-data-analysis/

> Data Visualization with Python and Matplotlib:
https://pythonprogramming.net/matplotlib-intro-tutorial/

Quandl (2020). Quandl API: Core Financial Data
https://www.quandl.com/

Quandl (2020). Get Financial Data Directly into Python.
https://www.quandl.com/tools/python

Vaidyanathan, V. (2020). Coursera: Investment Management with Python and Machine Learning Specialization
https://www.coursera.org/specializations/investment-management-python-machine-learning?utm_source=gg&utm_medium=sem&utm_content=01-CatalogDSA-Fin1-US&campaignid=9918777773&adgroupid=101741958718&device=c&keyword=&matchtype=b&network=g&devicemodel=&adpostion=&creativeid=433034295768&hide_mobile_promo&gclid=CjwKCAjw34n5BRA9EiwA2u9k3wMFe9XhXvmcdtRRu0_0jptT16ZSI0AUGJzWNHiPYfXN7StyPTS_mBoCRWcQAvD_BwE

Vaidyanathan, V. (2020b). Coursera: Course 1 - Introduction to Portfolio Construction and Analysis with Python
https://www.coursera.org/learn/introduction-portfolio-construction-python

Vaidyanathan, V. (2020c). Coursera: Course 2 - Advanced Portfolio Construction and Analysis with Python
https://www.coursera.org/learn/advanced-portfolio-construction-python

Vaidyanathan, V. (2020d). Coursera: Course 3 - Python and Machine Learning for Asset Management
https://www.coursera.org/learn/python-machine-learning-for-investment-management

Vaidyanathan, V. (2020e). Coursera: Course 4 - PYthon and Machine Learning for Asset Management with Alternative Data Sets
https://www.coursera.org/learn/machine-learning-asset-management-alternative-data