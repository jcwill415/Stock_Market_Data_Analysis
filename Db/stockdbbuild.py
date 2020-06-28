# Build DB for stock_dfs in MySQL

import pymysql
import pandas as pd
import sys

def csv_to_mysql(load_sql,host,user,password):

    "
    This function load to csv file to MySQL table according to the load_sql statement.
    "

    try:
        con = pymysql.connect(host = host,
                              user = user,
                              password = password,
                              autocommit = True,
                              local_infile = 1)
        print('Connected to DB:{}'.format(host))
        # Create cursor and execute Load SQL
        cursor = con.cursor()
        cursor.execute(load_sql)
        print('Successfull loaded the table from csv.')
        con.close()

    except Exception as e:
        print('Error:{}'.format(str(e)))
        sys.exit(1)

# Execution Example
load_sql = "LOAD DATA LOCAL INFILE 'c:/Users/JCW/Desktop/Stock_Market_Data_Analysis/stock_dfs' INTO TABLE user
