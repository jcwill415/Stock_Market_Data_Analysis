import pickle
def compile_data():
        favorite_color = pickle.load(open)("save.p", "rb")
        tickers = pickle.load
# Begin dataframe
        main_df = pd.DataFrame()

# Count in SP500 tickers list
        for count, ticker in enumerate(tickers):
                df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
                df.set_index('Date', inplace = True)

                df.rename(columns = {'Adj Close': ticker}, inplace = True)
                df.drop(['High', 'Low', 'Open', 'Volume'], 1, inplace = True)

# Joining dataframes together
                if main_df.empty:
                        main_df = df
                else:
                        main_df = main_df.join(df, how = 'outer')
        if count % 10 == 0:
                print(count)

        print(main_df.head())
        main_df.to_csv('sp500_joined_closes.csv')
compile_data()
