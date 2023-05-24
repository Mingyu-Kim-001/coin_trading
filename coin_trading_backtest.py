import os
from binance.client import Client
import pandas as pd
import numpy as np
import requests
from alpha_collection import Alphas
from utils import *

class market_neutral_trading_backtest_binance:
    def __init__(self):
        self.api_key = os.environ.get('BINANCE_API_KEY')
        self.api_secret = os.environ.get('BINANCE_API_SECRET')
        self.client = Client(self.api_key, self.api_secret)
        self.initial_budget = 10000
        self.trade_ratio = 0.1
        self.fee_rate = 0.00016 # 0.016%

    def get_binance_klines_data(self, symbol:str, interval:str='1d', limit:str='10000', **kwargs):
        api_endpoint = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, 'limit': limit, **kwargs}
        for _ in range(5):
            response = requests.get(api_endpoint, params=params)
            if response.status_code != 200:
                continue
            else:
                break
        else:
            raise Exception('API Error')

        klines = response.json()
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                         'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                         'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def get_binance_klines_data_1d(self, symbol, start_date='2017-01-01', end_date='2022-05-01', is_future=False):
        with open(f'./coin_backdata_daily/{"f" + symbol if is_future else symbol}.csv', 'r') as f:
            df = pd.read_csv(f)
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        df_date_dummy = pd.DataFrame({'date': pd.date_range(start_date, end_date, freq='1d')})
        df_extended = df_date_dummy.merge(df, on='date', how='left').fillna(0)
        return df_extended

    def backtest_coin_strategy(self, df_rank, df_date, df_close, symbols, stop_loss=-0.05):
        df_weight = self.standardize_rank(df_rank)
        df_agg = pd.concat([df_date, df_close, df_weight], axis=1)
        df_agg['return'] = sum([df_agg[f'{symbol}_close'].pct_change().fillna(0).mul(df_agg[f'{symbol}_weight'], axis=0).clip(lower=stop_loss) for symbol in symbols])
        df_agg['trade_size'] = df_agg[[f'{symbol}_weight' for symbol in symbols]].diff().abs().sum(1)
        df_agg['fee'] = df_agg['trade_size'].mul(self.fee_rate)
        df_agg['return_net'] = (1 - df_agg['fee']) * (1 + df_agg['return']) - 1
        df_agg['cumulative_fee'] = df_agg['fee'].cumsum()
        df_agg['cumulative_return'] = (1 + df_agg['return_net']).cumprod()
        return df_agg

    def standardize_rank(self, df_rank:pd.DataFrame):
        df_rank_mean = df_rank.mean(1)
        df_rank_not_nan_cnt = df_rank.notna().sum(1)
        df_rank_odd = df_rank_not_nan_cnt % 2
        df_rank_normalizer = (df_rank_not_nan_cnt // 2) ** 2 * (1 - df_rank_odd) + (df_rank_not_nan_cnt // 2) * (
              df_rank_not_nan_cnt // 2 + 1) * df_rank_odd
        df_standardized_weight = df_rank.sub(df_rank_mean, axis=0).div(df_rank_normalizer, axis=0).fillna(0)
        return df_standardized_weight

if __name__ == '__main__':
    import datetime
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    backtest = market_neutral_trading_backtest_binance()
    alpha_collection = Alphas()
    alpha_list = alpha_collection.alpha_list
    dict_df_klines = {}
    start_date = datetime.date(2017, 8, 17)
    end_date = datetime.date(2022, 5, 1)
    print(f'past {start_date} ~ {end_date}')
    symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'DOGEUSDT', 'LTCUSDT', 'MATICUSDT', 'TRXUSDT', 'ADAUSDT', 'SOLUSDT']
    for symbol in symbols:
        dict_df_klines[symbol] = backtest.get_binance_klines_data_1d(symbol, start_date, end_date)
    df_date = list(dict_df_klines.values())[0]['date']
    df_close = pd.concat([df_klines['close'].astype('float').rename(f'{symbol}_close') for symbol, df_klines in dict_df_klines.items()], axis=1)
    for alpha in alpha_list:
        df_rank = getattr(alpha_collection, alpha)(dict_df_klines)
        final_return = backtest.backtest_coin_strategy(df_rank, df_date, df_close, symbols)['cumulative_return'].iloc[-1]
        print(alpha, 'final', round(final_return, 2))


    start_date = datetime.date(2022, 5, 2)
    end_date = datetime.date(2023, 5, 1)
    is_future = True
    print(f'recent {start_date} ~ {end_date}')
    symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'DOGEUSDT', 'LTCUSDT', 'MATICUSDT', 'TRXUSDT', 'ADAUSDT', 'SOLUSDT']
    for symbol in symbols:
        dict_df_klines[symbol] = backtest.get_binance_klines_data_1d(symbol, start_date, end_date, is_future)
    df_date = list(dict_df_klines.values())[0]['date']
    df_close = pd.concat([df_klines['close'].astype('float').rename(f'{symbol}_close') for symbol, df_klines in dict_df_klines.items()], axis=1)
    for alpha in alpha_list:
        df_rank = getattr(alpha_collection, alpha)(dict_df_klines)
        final_return = backtest.backtest_coin_strategy(df_rank, df_date, df_close, symbols)['cumulative_return'].iloc[-1]
        print(alpha, 'final', round(final_return, 2))






