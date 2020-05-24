import urllib
import json

html = urlopen('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = BeautifulSoup(html,'lxml')
sp500_raw = soup.find('table', {'class': 'wikitable sortable'})

spsymbol =[]

for row in sp500_raw.findAll('tr')[1:]:
    spsymbols = row.findAll('td')[0].text.strip()
    spsymbol.append(spsymbols)

start = datetime.date(2000,1,1)
end = datetime.date.today()

df = pd.DataFrame(spsymbol, columns=['Stock_name'])
df.Stock_name.str.replace('^\d+ +','').to_csv('SP Scrapped data.csv', index=False, header=False) 

stock_data = []
with open('SP Scrapped data.csv') as csvfile:
    stockticker_data = csv.reader(csvfile, delimiter=' ')
    for stockticker_data1 in stockticker_data:
        stockticker_data1 = [col.replace('.', '-') for col in stockticker_data1]
        for row in stockticker_data1:
            print(row)
            all_data = []
            for ticker in row:
                try:
                    stock_data.append(web.get_data_yahoo(ticker, start, end))
                    for df in stock_data:
                        df.to_csv(ticker, header=True, index=True, columns=['Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'], sep=' ')
                except Exception as ex:
                    print('Ex:', ex)
