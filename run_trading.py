import os
from datetime import datetime, timedelta

import pandas as pd
from binance.client import Client
import asyncio
from binance import AsyncClient, BinanceSocketManager, ThreadedWebsocketManager

import alpha_collection
from utils import *


api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_SECRET_API_KEY')
client = Client(api_key, api_secret)

def get_binance_klines_data(timestamp_start, timestamp_end, symbol, interval='1m'):
    timestamp_start_str = str(int(timestamp_start.timestamp()))
    timestamp_end_str = str(int(timestamp_end.timestamp()))
    klines = client.futures_historical_klines(symbol, interval, timestamp_start_str, timestamp_end_str)
    df = pd.DataFrame(klines,
                      columns=['open_time', 'open', 'high', 'low', 'close', 'volume',
                               'close_time',
                               'quote_asset_volume',
                               'number_of_trades', 'taker_buy_base_asset_volume',
                               'taker_buy_quote_asset_volume', 'ignore'])
    df['open_time'] = df['open_time'].apply(lambda x: datetime.fromtimestamp(x / 1000))
    df['close_time'] = df['close_time'].apply(lambda x: datetime.fromtimestamp(x / 1000))
    return df

# def get_next_position(self, dict_df_klines, alpha, quote_total_size):
#     df_weight = alpha(dict_df_klines)
#     df_next_position = df_weight.iloc[-1:, :] * quote_total_size
#     df_next_position.columns = [symbol_weight.split('_')[0] for symbol_weight in df_next_position.columns] #remove _weight
#     return df_next_position

# def simple_momentum_3(self):


def get_current_binance_futures_balance():
    #get my current binance futures amount
    futures_account = client.futures_account()
    df_futures_account = pd.DataFrame(futures_account['assets'])
    for column_name in df_futures_account.columns:
        if column_name not in ['asset', 'marginAvailable', 'updateTime']:
            df_futures_account[column_name] = df_futures_account[column_name].astype('float')
    return df_futures_account

def get_current_future_price(symbol):
    futures_ticker = client.futures_ticker(symbol=symbol)
    return float(futures_ticker['lastPrice'])

def create_order(symbol, price, quantity):
    order = client.futures_create_order(
        symbol=symbol,
        type="LIMIT",
        side="BUY",
        timeInForce='FOK',
        price=price,
        quantity=quantity
    )

def get_futures_trading_rules():
    with open('./futures_trading_rules/futures_trading_rules.csv', 'r') as f:
        df_future_trading_rules = pd.read_csv(f)
    return df_future_trading_rules

def get_futures_order_book(symbol):
    order_book = client.futures_order_book(symbol=symbol)
    return order_book

# def order_real_time(self):




if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    total_quantity = 300
    symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'DOGEUSDT', 'LTCUSDT', 'MATICUSDT', 'TRXUSDT', 'ADAUSDT', 'SOLUSDT']
    close_72hours = {}
    dict_df_klines = {}
    df_current_futures_balance = get_current_binance_futures_balance()
    close_72hours = {symbol: float(client.futures_historical_klines(symbol, '1h', '72 hours ago UTC')[0][4]) for symbol in symbols}
    current_price = {symbol: float(client.futures_ticker(symbol=symbol)['lastPrice']) for symbol in symbols}
    df_price = pd.concat([pd.DataFrame(close_72hours, index=[0]), pd.DataFrame(current_price, index=[1])], axis=0)
    df_weight = neutralize_weight(df_price.pct_change().loc[[1]])
    df_quantity = (df_weight * total_quantity)
    df_quantity_and_price = pd.concat([df_quantity, pd.DataFrame(current_price, index=['price'])], axis=0)
    df_quantity_and_price_trimmed = trim_quantity(df_quantity_and_price)
    print(df_quantity_and_price_trimmed)


