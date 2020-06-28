# Using Interactive Brokers API for Automated Trading
import os
import sys
import time
import numpy as np

import ibAPI as ib
from PyQt5.QtWidgets import (QWidget, QLineEdit, QGridLayout, QLabel, QApplication)
import pyqtgraph as pg
#from ib_insync import *

#from ib.ext.Contract import Contract
#from ib.ext.Order import Order
#from ib.opt import Connect, message

util.useQt()

ib = IB()
ib.connect('127.0.0.1', 7497, clientId = 986)

#from ib.opt import Connection, message
#from ib.ext.Contract import Contract
#from ib.ext.Order import Order

# Setup contract and place order
# Define variable make_contract, you need parameters:
# stock symbol, security type, which exchange, under which primary exchange, & currency
def make_contract(symbol, sec_type, exch, prim_exchange, curr):
    Contract.m_symbol = symbol
    Contract.m_secType = sec_Type
    Contract.m_exchange = exch
    Contract.m_primaryExch = prim_exch
    Contract.m_currency = curr
    return Contract

# Make Order
# Market order (instant), or limit order
# action = buy or sell, quantity = how many shares, price = how much to pay per share
def make_order(action, quantity, price = None):
    # if price is not None, we will specify a limit price
    if price is not None:
        order = Order()
        order.m_orderType = 'LMT'
        order.m_totalQuantity = quantity
        order.m_action = action
        order.m_lmtPrice = price

    # if we do not specify limit price, we have a market order
    else:

        order = Order()
        order.m_orderType = 'MKT'
        order.m_totalQuantity = quantity
        order.m_action = action

    return order

#def main():
#    conn = Connection.create(port = 7497, clientId = 986)
#    conn.connect()

# Here we will place order for Microsoft (ticker = MSFT) stock (STK)
# Using SMART exchange = Interactive Brokers smart routing
# SMART automatically selects based on price and availability
# Currency = US Dollars
    oid = 1
    cont = make_contract('CSCO', 'STK', 'SMART', 'SMART', 'USD')

    # Offer to buy one share at limit price $200.00
    offer = make_offer('BUY', 1, 200)
    conn.placeOrder(oid, cont, offer)
    conn.disconnect()
IB()
