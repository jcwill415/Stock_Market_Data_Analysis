# Imports for trading algorithm

import sys
import socket
import struct
import array
import inspect
import time
import argparse
import os.path
import json
import csv
from pprint import pprint
import twitter
from fred import Fred
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.utils import formataddr

import urllib
#import urllib2
import requests
from bs4 import BeautifulSoup as bs
import shutil

from ibapi.wrapper import EWrapper
import ibapi.decoder
import ibapi.wrapper
from ibapi.common import *
from ibapi.ticktype import TickType, TickTypeEnum
from ibapi.comm import *
from ibapi.message import IN, OUT
from ibapi.client import EClient
from ibapi.connection import Connection
from ibapi.reader import EReader
from ibapi.utils import *
from ibapi.execution import ExecutionFilter
from ibapi.scanner import ScannerSubscription
from ibapi.order_condition import *
from ibapi.contract import *
from ibapi.order import *
from ibapi.order_state import *
from Testbed.ContractSamples import ContractSamples
from Testbed.OrderSamples import OrderSamples

import datetime
import datetime as dt
from datetime import timedelta
import quandl as qdl
# Install https://www.anaconda.com/download/ 
import pandas_datareader as pdr
import pandas as pd
# Import Matplotlib's `pyplot` module as `plt`
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
#import statsmodels.api as sm
# Import the `datetools` module from `pandas`
#from pandas.core import datetools
import pandas.tseries as pdts

# Create Program. Create TestApp class
# Includes Client & Wrapper classes
# Try to place test order

# Create placeOneOrder function
# Use OrderSamples to create a BUY order for 100 shares
# Select the account, in this case account DU2324971 
# NOTE: Do not ccnnect to live account, use Paper Account
# call the Client function placeOrder with a sample USStock.
# The ContractSamples.USStock() returns a test stock.
# Create Market Order for stock

class TestApp(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self.nextValidOrderId = 10
		
@iswrapper
def nextValidId(self, orderId:int):
    super().nextValidId(orderId)
    logging.debug("setting nextValidOrderId: %d", orderId)
    self.nextValidOrderId = orderId

def placeOneOrder(self):
    #self.simplePlaceOid = self.nextOrderId()
    faOrderOneAccount = OrderSamples.MarketOrder("BUY", 100)
    faOrderOneAccount.account = "DU2324971"
    self.placeOrder(self.nextOrderId(), ContractSamples.USStock(), faOrderOneAccount)
		
def main():
    #put your calls in here
    if __name__ == "__main__":
        main()

# Contine with additional wrapper functions to get order information such as order status.
# Add some more variables in the init function.

class TestApp(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self.nextValidOrderId = 10
        self.simplePlaceOid = None
        self.permId2ord = {}
        self.hData = {}
        self.hDataColumns = ['Date','Open','High','Low','Close','Volume','BarCount','Average']
        self.hDataIndex = []
        self.hDataRecords = []
        self.lineCount = 0
        self.hDataCurrent = False
        self.hDataMonthly = False
        self.cDataPrice = 0
        self.currentSymbol = "SPY"
		
    @iswrapper
    def nextValidId(self, orderId:int):
        super().nextValidId(orderId)
        logging.debug("setting nextValidOrderId: %d", orderId)
        self.nextValidOrderId = orderId

    def placeOneOrder(self):
        #self.simplePlaceOid = self.nextOrderId()
        faOrderOneAccount = OrderSamples.MarketOrder("BUY", 100)
        faOrderOneAccount.account = "DU9000000"
        self.placeOrder(self.nextOrderId(), ContractSamples.USStock(), faOrderOneAccount)

    def getWebFile(self, URL, fileName):
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        local_filename = fileName
        pdf_request = requests.get(URL + local_filename, headers=headers, stream=True)
        with open('data/' + local_filename, 'wb') as f:
            shutil.copyfileobj(pdf_request.raw, f)
			
    @iswrapper
    def error(self, *args):
        super().error(*args)
        print(current_fn_name(), vars())

    @iswrapper
    def winError(self, *args):
        super().error(*args)
        print(current_fn_name(), vars())

    @iswrapper
    def openOrder(self, orderId:OrderId, contract:Contract, order:Order, 
                  orderState:OrderState):
        super().openOrder(orderId, contract, order, orderState)
        print(current_fn_name(), vars())

        order.contract = contract
        self.permId2ord[order.permId] = order

    @iswrapper
    def openOrderEnd(self, *args):
        super().openOrderEnd()
        logging.debug("Received %d openOrders", len(self.permId2ord))

    @iswrapper
    def orderStatus(self, orderId:OrderId , status:str, filled:float,
                    remaining:float, avgFillPrice:float, permId:int, 
                    parentId:int, lastFillPrice:float, clientId:int, 
                    whyHeld:str):
        super().orderStatus(orderId, status, filled, remaining,
            avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld)

    @iswrapper
    def tickPrice(self, tickerId: TickerId , tickType: TickType, price: float, attrib):
        super().tickPrice(tickerId, tickType, price, attrib)
        print(current_fn_name(), tickerId, TickTypeEnum.to_str(tickType), price, attrib, file=sys.stderr)
        #tickPrice 1001 LAST 264.42 0,0
        if(TickTypeEnum.to_str(tickType) == 'LAST'):
            self.cDataPrice = price

    @iswrapper
    def tickSize(self, tickerId: TickerId, tickType: TickType, size: int):
        super().tickSize(tickerId, tickType, size)
        print(current_fn_name(), tickerId, TickTypeEnum.to_str(tickType), size, file=sys.stderr)

    @iswrapper
    def tickSnapshotEnd(self, reqId: int):
        super().tickSnapshotEnd(reqId)
        print("TickSnapshotEnd:", reqId)
        if(self.isConnected()):
            self.disconnect()

    @iswrapper
    def realtimeBar(self, reqId:TickerId, time:int, open:float, high:float, low:float, close:float, volume:int, wap:float, count:int):
        super().realtimeBar(reqId, time, open, high, low, close, volume, wap, count)
        print("RealTimeBars. ", reqId, ": time ", time, ", open: ",open, ", high: ", high, ", low: ", low, ", close: ", close, ", volume: ", volume,", wap: ", wap, ", count: ", count)

    @iswrapper
    def scannerParameters(self, xml:str):
        open('scanner.xml', 'w').write(xml)

    @iswrapper
    def position(self, account: str, contract: ibapi.contract.Contract, position: float, avgCost: float):
        print('Position: {} {} @ {}'.format(position, contract.symbol, avgCost))

    @iswrapper
    def positionEnd(self):
        pass
    
    def fundamentalData(self, reqId: TickerId, data: str):
        print("FundamentalData. ", reqId, data)

    def newsProviders(self, newsProviders: ListOfNewsProviders):
        print("newsProviders: ")
        for provider in newsProviders:
            print(provider)
    
    def historicalData(self, reqId:int, bar: BarData):
        row = []
        row.append(str(bar.date))
        row.append(bar.open)
        row.append(bar.high)
        row.append(bar.low)
        row.append(bar.close)
        row.append(bar.volume)
        row.append(bar.barCount)
        row.append(bar.average)
        self.hDataRecords.append(row)
        self.hDataIndex.append(self.lineCount)
        self.lineCount += 1
        print("HistoricalData. ", reqId, " ,Date:", bar.date, ",Open:", bar.open,",High:", bar.high, ",Low:", bar.low, ",Close:", bar.close, ",Volume:", bar.volume,",Count:", bar.barCount, ",WAP:", bar.average)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        super().historicalDataEnd(reqId, start, end)
        self.hData["columns"] = self.hDataColumns
        self.hData["index"] = self.hDataIndex
        self.hData["data"] = self.hDataRecords

        #Format:
        #jsonOut = {"columns":["Date","Open","High"], 
        #"index":[0, 1],
        #"data": [["2017-12-18", 268.08, 268.67], ["2017-12-19", 268.48, 268.53]]}
        if(self.currentSymbol == 'SPY'):
            current_file = 'data/spy_current.json'
            month_file = 'data/spy_month.json'
            all_file = 'data/spy.json'
            all_file_csv = 'data/spy.csv'
        elif(self.currentSymbol == 'IWM'):
            current_file = 'data/iwm_current.json'
            month_file = 'data/iwm_month.json'
            all_file = 'data/iwm.json'
            all_file_csv = 'data/iwm.csv'

        jsonDump = json.dumps(self.hData)
        if(self.hDataCurrent):
            #write the current file
            with open(current_file, 'w') as f:
                f.write(jsonDump)
            self.hDataCurrent = False
        elif(self.hDataMonthly):
            with open(month_file, 'w') as f:
                f.write(jsonDump)
            self.hDataMonthly = False
        else:
            with open(all_file, 'w') as f:
                #writer = csv.writer(f, delimiter=',', lineterminator='\r\n', quotechar="'")
                f.write(jsonDump)
            w = csv.writer(open(all_file_csv, "wt", newline=''), quoting=csv.QUOTE_NONE, escapechar=' ', quotechar='')
            #w = csv.writer(fw, delimiter=',', lineterminator='\r\n', quotechar="'")

            for item in self.hData.items():
                w.writerow([item[1]])
        #clear hData
        self.hData.clear()
        self.hDataRecords.clear()
        self.hDataIndex.clear()
        self.lineCount = 0

        print("HistoricalDataEnd ", reqId, "from", start, "to", end)
        if(self.isConnected()):
            self.disconnect()

    def historicalDataUpdate(self, reqId: int, bar: BarData):
        print("HistoricalDataUpdate. ", reqId, " Date:", bar.date, "Open:", bar.open,"High:", bar.high, "Low:", bar.low, "Close:", bar.close, "Volume:", bar.volume,"Count:", bar.barCount, "WAP:", bar.average)

def main():
    #put your calls in here
    if __name__ == "__main__":
        main()
	
# FURTHER STEPS
# Now we need to create data visualizations, such as charts
# Follow the example for momentum described in my previous blog post.



