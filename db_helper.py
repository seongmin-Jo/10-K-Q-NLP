import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import requests

from ratelimit import limits, sleep_and_retry


class SecAPI(object):
    SEC_CALL_LIMIT = {'calls': 10, 'seconds': 1}

    @staticmethod
    @sleep_and_retry
    # Dividing the call limit by half to avoid coming close to the limit
    @limits(calls=SEC_CALL_LIMIT['calls'] / 2, period=SEC_CALL_LIMIT['seconds'])
    def _call_sec(url):
        return requests.get(url)

    def get(self, url):
        return self._call_sec(url).text

def get_sp500():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    doc = BeautifulSoup(response.text, 'html.parser')
    symbol = pd.Series([data.text.replace('\n', '').replace('.','-') for data in doc.select(
        '#constituents > tbody > tr > td:nth-child(1)')], name='Symbol')
    security = pd.Series([data.text.replace('\n', '') for data in doc.select(
        '#constituents > tbody > tr > td:nth-child(2)')], name='Security')
    gics_sector = pd.Series([data.text.replace('\n', '') for data in doc.select(
        '#constituents > tbody > tr > td:nth-child(4)')], name='GICS')
    ticker = pd.Series([data.text.replace('\n', '') for data in doc.select(
        '#constituents > tbody > tr > td:nth-child(8)')], name='CIK')
    result = pd.concat([symbol, security, gics_sector, ticker], axis=1)
    #result['marketcap'] = result['Symbol'].map(lambda x: yf.Ticker(x).info['marketCap'])
    return result
    
def get_cik_by_sector(sector):
    temp_df = get_sp500().pivot('Symbol', 'GICS', 'CIK').unstack().reset_index()
    temp_df.columns = ['GICS Sector', 'Symbol', 'CIK']
    cik_by_sector = temp_df.dropna().reset_index().drop('index', axis=1)
    return cik_by_sector[cik_by_sector['GICS Sector'] == sector][['Symbol', 'CIK']].reset_index(drop=True)
    
    
import yfinance as yf

def daily_returns(ticker, return_marketcap=True, include_return=False):
    stock = yf.Ticker(ticker)
    df = stock.history(period='max')
    df.columns = [text.lower() for text in df.columns]
    df = df.rename(columns={'close':ticker})
    df[ticker+'_return'] = df[ticker].pct_change()
    result = df[[ticker]]
    if include_return:
        result = df[[ticker,ticker+'_return']]
    if return_marketcap:
        result = (stock.info['marketCap'], result)
    return result

def returns_for_tickers(list_of_tickers, return_marketcap=True, include_return=False):
    ticker_daily_return_list = []
    marketcap_list = []
    for ticker in list_of_tickers:
        try:
            if return_marketcap:
                marketcap, temp_daily_returns = daily_returns(ticker, return_marketcap=return_marketcap, include_return=include_return)
                marketcap_list.append(marketcap)
            else:
                temp_daily_returns = daily_returns(ticker, return_marketcap=return_marketcap)
        except:
            print("Could not get ticker for {}".format(ticker))
            continue
        ticker_daily_return_list.append(temp_daily_returns)
        
    return marketcap_list, pd.concat(ticker_daily_return_list, axis=1)

from fake_useragent import UserAgent

def get_etf_code():
    
    etf_data = {'Item': ['TIGER 200 IT', 'TIGER 헬스케어', 'TIGER 200 금융', 'TIGER 증권', 'TIGER 200 생활소비재'],
            'Code': [139260, 143860, 139270, 157500, 227560]}
    etf_data = pd.DataFrame(etf_data)
    day_stock_url = 'https://finance.naver.com/item/sise_day.nhn?code={}'
    etf_data['Day_URL'] = etf_data["Code"].map(lambda x: day_stock_url.format(x))
    return etf_data


def get_etf_data(code):
    day_stock_url = 'https://finance.naver.com/item/sise_day.nhn?code={}'.format(code)
    headers = {
        'user-agent': UserAgent().chrome
    }
    url = day_stock_url + '&page={}'
    URL = url.format(1)
    response = requests.get(URL, headers=headers)
    doc = BeautifulSoup(response.text, 'html5lib')
    header = doc.select(
        'body > table.type2 > tbody > tr:nth-child(1)')[0].text.replace('\n', '').split('\t')
    header = [row for row in header if len(row) >= 1]
    n = 0
    data = []
    try:
        while True:
            n += 1
            URL = url.format(n)
            response = requests.get(URL, headers=headers)
            doc = BeautifulSoup(response.text, 'html5lib')

            for idx, row in enumerate([tah.text.replace('\t', '').replace('\n', '') for tah in doc.select('.tah')]):
                if idx % 7 == 0:
                    temp = []
                    temp.append(row)
                elif idx % 7 == 6:
                    temp.append(int(row.replace(',','')))
                    data.append(temp)
                else:
                    temp.append(int(row.replace(',','')))
                if '맨뒤' not in doc.text:
                    #print('Page {} Done!'.format(n))
                    df = pd.DataFrame(data, columns=header)
                    df = df.set_index('날짜')
                    df.index = pd.to_datetime(df.index)
                    df.index.name = 'Date'
                    return df
    except:
        df = pd.DataFrame(data, columns=header)
        df = df.set_index('날짜')
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        return df

def print_ten_k_data(ten_k_data, fields, field_length_limit=50):
    indentation = '  '

    print('[')
    for ten_k in ten_k_data:
        print_statement = '{}{{'.format(indentation)
        for field in fields:
            value = str(ten_k[field])

            # Show return lines in output
            if isinstance(value, str):
                value_str = '\'{}\''.format(value.replace('\n', '\\n'))
            else:
                value_str = str(value)

            # Cut off the string if it gets too long
            if len(value_str) > field_length_limit:
                value_str = value_str[:field_length_limit] + '...'

            print_statement += '\n{}{}: {}'.format(indentation * 2, field, value_str)

        print_statement += '},'
        print(print_statement)
    print(']')


def plot_similarities(similarities_list, dates, title, labels):
    assert len(similarities_list) == len(labels)

    plt.figure(1, figsize=(10, 7))
    for similarities, label in zip(similarities_list, labels):
        plt.title(title)
        plt.plot(dates, similarities, label=label, marker='o')
        plt.legend()
        plt.xticks(dates)
        plt.xticks(rotation=90)

    plt.show()
