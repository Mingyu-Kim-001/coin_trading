import os
from datetime import datetime, timedelta

import pandas as pd
from binance.client import Client
import asyncio
from binance import AsyncClient, BinanceSocketManager

import alpha_collection
from utils import *


class Trading():
    def __init__(self):
        self.BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
        self.BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_API_KEY')
        self.client = Client(self.BINANCE_API_KEY, self.BINANCE_SECRET_KEY)
        self.df_future_trading_rules = None

    def get_binance_klines_data(self, timestamp_start, timestamp_end, symbol, interval='1m'):
        timestamp_start_str = str(int(timestamp_start.timestamp()))
        timestamp_end_str = str(int(timestamp_end.timestamp()))
        klines = self.client.futures_historical_klines(symbol, interval, timestamp_start_str, timestamp_end_str)
        df = pd.DataFrame(klines,
                          columns=['open_time', 'open', 'high', 'low', 'close', 'volume',
                                   'close_time',
                                   'quote_asset_volume',
                                   'number_of_trades', 'taker_buy_base_asset_volume',
                                   'taker_buy_quote_asset_volume', 'ignore'])
        df['open_time'] = df['open_time'].apply(lambda x: datetime.fromtimestamp(x / 1000))
        df['close_time'] = df['close_time'].apply(lambda x: datetime.fromtimestamp(x / 1000))
        return df

    def get_next_position(self, dict_df_klines, alpha, quote_total_size):
        df_weight = alpha(dict_df_klines)
        df_next_position = df_weight.iloc[-1:, :] * quote_total_size
        df_next_position.columns = [symbol_weight.split('_')[0] for symbol_weight in df_next_position.columns] #remove _weight
        return df_next_position

    def get_current_binance_futures_balance(self):
        #get my current binance futures amount
        futures_account = self.client.futures_account()
        df_futures_account = pd.DataFrame(futures_account['assets'])
        for column_name in df_futures_account.columns:
            if column_name not in ['asset', 'marginAvailable', 'updateTime']:
                df_futures_account[column_name] = df_futures_account[column_name].astype('float')

    def get_current_future_price(self, symbol):
        futures_ticker = self.client.futures_ticker(symbol=symbol)
        return float(futures_ticker['lastPrice'])

    def create_order(self, symbol, price, quantity):
        order = self.client.futures_create_order(
            symbol=symbol,
            type="LIMIT",
            side="BUY",
            timeInForce='GTC',
            price=price,
            quantity=quantity
        )

    def get_futures_trading_rules(self):
        with open('./futures_trading_rules/futures_trading_rules.csv', 'r') as f:
            self.df_future_trading_rules = pd.read_csv(f)
        return self.df_future_trading_rules

    def get_futures_order_book(self, symbol):
        client = Client(self.BINANCE_API_KEY, self.BINANCE_SECRET_KEY)
        order_book = client.futures_order_book(symbol=symbol)
        return order_book


if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    trading = Trading()
    trading.get_futures_trading_rules()
    n_days = 4
    alphas = alpha_collection.Alphas()
    now = datetime.now()
    timestamp_interested = [now - timedelta(days=day) for day in range(n_days-1, -1, -1)]
    symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'DOGEUSDT', 'LTCUSDT', 'MATICUSDT', 'TRXUSDT', 'ADAUSDT', 'SOLUSDT']
    dict_df_klines = {}
    for symbol in symbols:
        df_klines = pd.concat([trading.get_binance_klines_data(timestamp - timedelta(minutes=1), timestamp, symbol, interval='1m') for timestamp in timestamp_interested])
        df_klines = df_klines.append(pd.DataFrame([[None] * len(df_klines.columns)], columns=df_klines.columns), ignore_index=True) #add null rows
        dict_df_klines[symbol] = df_klines
    alpha = lambda x: alphas.close_momentum_nday(x, 3)
    next_position = trading.get_next_position(dict_df_klines, alpha, quote_total_size=30)
    print(next_position)


