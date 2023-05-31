import os

import numpy as np
import requests
from binance.client import Client

from utils import *


class market_neutral_trading_backtest_binance:
    def __init__(self):
        self.api_key = os.environ.get('BINANCE_API_KEY')
        self.api_secret = os.environ.get('BINANCE_API_SECRET')
        self.client = Client(self.api_key, self.api_secret)
        self.initial_budget = 10000
        self.trade_ratio = 0.1
        self.fee_rate = 0.00016  # 0.016%

    def get_binance_klines_data(self, symbol: str, interval: str = '1d', limit: str = '10000', **kwargs):
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
        for column in ['open', 'high', 'low', 'close', 'volume']:
            df_extended[column] = df_extended[column].astype(float)
        return df_extended

    def backtest_coin_strategy(self, df_weight, dict_df_klines, df_date, symbols, stop_loss=-1):
        df_neutralized_weight = self.neutralize_weight(df_weight)
        dict_df_return, dict_df_trade_size = {}, {}
        df_agg = pd.DataFrame(df_date, columns=['date'])
        for symbol in symbols:
            df_neutralized_weight_symbol = df_neutralized_weight[f'{symbol}_weight']
            pct_change = dict_df_klines[symbol]['close'].pct_change().fillna(0)
            daily_maximum_drawdown = (dict_df_klines[symbol]['open'] - dict_df_klines[symbol]['low']) / dict_df_klines[symbol]['open'] - 1
            daily_maximum_gain = (dict_df_klines[symbol]['high'] - dict_df_klines[symbol]['open']) / dict_df_klines[symbol]['open'] - 1
            is_stop_loss = ((df_neutralized_weight_symbol < 0) & (daily_maximum_drawdown < stop_loss)) \
                           | ((df_neutralized_weight_symbol > 0) & (daily_maximum_gain > -stop_loss))
            dict_df_return[symbol] = pd.Series(np.where(is_stop_loss,
                                                        -abs(stop_loss * df_neutralized_weight_symbol),
                                                        pct_change * df_neutralized_weight_symbol
                                                        ))
            df_neutralized_weight_symbol_lag = df_neutralized_weight_symbol.shift(1)

            dict_df_trade_size[symbol] = pd.Series(
                np.where(is_stop_loss,
                         np.where(is_stop_loss.shift(1),
                                  2 * abs(df_neutralized_weight_symbol),
                                  abs(df_neutralized_weight_symbol) + abs(
                                      df_neutralized_weight_symbol_lag - df_neutralized_weight_symbol)),
                         np.where(is_stop_loss.shift(1),
                                  abs(df_neutralized_weight_symbol),
                                  abs(df_neutralized_weight_symbol - df_neutralized_weight_symbol_lag)))
            )

        df_agg['return'] = pd.DataFrame(dict_df_return).sum(axis=1)
        df_agg['trade_size'] = pd.DataFrame(dict_df_trade_size).sum(axis=1)
        df_agg['fee'] = df_agg['trade_size'].mul(self.fee_rate)
        df_agg['return_net'] = (1 - df_agg['fee']) * (1 + df_agg['return']) - 1
        df_agg['cumulative_fee'] = df_agg['fee'].cumsum()
        df_agg['cumulative_return'] = (1 + df_agg['return_net']).cumprod()
        df_agg['possible_maximum_drawdown'] = self.get_possible_maximum_drawdown(df_agg['cumulative_return'])
        return df_agg

    def neutralize_weight(self, df_weight: pd.DataFrame):
        df_weight_mean = df_weight.mean(1)
        df_weight_neutralized = df_weight.sub(df_weight_mean, axis=0)
        df_weight_normalizer = df_weight_neutralized.abs().sum(1)
        df_weight_normalized = df_weight_neutralized.div(df_weight_normalizer, axis=0).fillna(0)
        return df_weight_normalized

    def get_possible_maximum_drawdown(self, df_cumulative_return):
        df_cumulative_max = df_cumulative_return.cummax()
        df_drawdown = df_cumulative_return / df_cumulative_max - 1
        return df_drawdown
