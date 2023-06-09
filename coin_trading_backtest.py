import os

import numpy as np
import pandas as pd
import requests
from utils import *
import datetime


class market_neutral_trading_backtest_binance:
    def __init__(self):
        self.initial_budget = 10000
        self.trade_ratio = 0.1
        self.fee_rate = 0.0002  # 0.02%

    def get_binance_klines_data_1d(self, symbol, start_date='2017-01-01', end_date='2022-05-01', is_future=False):
        with open(f'./coin_backdata_daily/{"f" + symbol if is_future else symbol}.csv', 'r') as f:
            df = pd.read_csv(f)
            df['timestamp'] = df['timestamp'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df_date_dummy = pd.DataFrame({'timestamp': pd.date_range(start_date, end_date, freq='1d')})
        df_extended = df_date_dummy.merge(df, on='timestamp', how='left').fillna(0)
        for column in ['open', 'high', 'low', 'close', 'volume']:
            df_extended[column] = df_extended[column].astype(float)
        if symbol == 'DOGEUSDT':
            df_extended = df_extended[df_extended['timestamp'] >= pd.to_datetime('2022-01-01')] # remove volatile data
        return df_extended

    def get_binance_klines_data_1h(self, symbol, start_datetime=datetime.datetime(2017, 1, 1, 9, 0, 0), end_datetime=datetime.datetime(2022, 6, 1, 0, 0, 0), freq='1h', is_future=False):
        with open(f'./coin_backdata_hourly/{"f" + symbol if is_future else symbol}.csv', 'r') as f:
            df = pd.read_csv(f)
            df['timestamp'] = df['timestamp'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df_date_dummy = pd.DataFrame({'timestamp': pd.date_range(start_datetime, end_datetime, freq=freq)})
        df_extended = df_date_dummy.merge(df, on='timestamp', how='left').fillna(0)
        for column in ['open', 'high', 'low', 'close', 'volume']:
            df_extended[column] = df_extended[column].astype(float)
        if symbol == 'DOGEUSDT':
            df_extended = df_extended[df_extended['timestamp'] >= pd.to_datetime('2022-01-01')] # remove volatile data
        return df_extended

    def backtest_coin_strategy(self, df_neutralized_weight, dict_df_klines, df_timestamp, symbols, stop_loss=-1, leverage=1):
        dict_df_return, dict_df_trade_size = {}, {}
        df_agg = pd.DataFrame(df_timestamp, columns=['timestamp'], index=df_neutralized_weight.index)
        for symbol in symbols:
            df_neutralized_weight_symbol = df_neutralized_weight[symbol]
            pct_change = dict_df_klines[symbol]['close'].pct_change().fillna(0).replace([-np.inf, np.inf], 0)
            daily_maximum_drawdown = (dict_df_klines[symbol]['open'] - dict_df_klines[symbol]['low']) / dict_df_klines[symbol]['open'] - 1
            daily_maximum_gain = (dict_df_klines[symbol]['high'] - dict_df_klines[symbol]['open']) / dict_df_klines[symbol]['open'] - 1
            # is_stop_loss = ((df_neutralized_weight_symbol < 0) & (daily_maximum_drawdown < stop_loss)) \
            #                | ((df_neutralized_weight_symbol > 0) & (daily_maximum_gain > -stop_loss))
            # dict_df_return[symbol] = pd.Series(np.where(is_stop_loss,
            #                                             -abs(stop_loss * df_neutralized_weight_symbol),
            #                                             pct_change * df_neutralized_weight_symbol
            #                                             ))
            dict_df_return[symbol] = pct_change * df_neutralized_weight_symbol
            df_neutralized_weight_symbol_lag = df_neutralized_weight_symbol.shift(1)

            # dict_df_trade_size[symbol] = pd.Series(
            #     np.where(is_stop_loss,
            #              np.where(is_stop_loss.shift(1),
            #                       2 * abs(df_neutralized_weight_symbol),
            #                       abs(df_neutralized_weight_symbol) + abs(
            #                           df_neutralized_weight_symbol_lag - df_neutralized_weight_symbol)),
            #              np.where(is_stop_loss.shift(1),
            #                       abs(df_neutralized_weight_symbol),
            #                       abs(df_neutralized_weight_symbol - df_neutralized_weight_symbol_lag)))
            # )
            dict_df_trade_size[symbol] = abs(df_neutralized_weight_symbol - df_neutralized_weight_symbol_lag)

        df_agg['return'] = (pd.DataFrame(dict_df_return).sum(axis=1) * leverage).clip(-1, float("inf"))
        df_agg['trade_size'] = pd.DataFrame(dict_df_trade_size).sum(axis=1) * leverage
        df_agg['fee'] = df_agg['trade_size'].mul(self.fee_rate)
        df_agg['return_net'] = (1 - df_agg['fee']) * (1 + df_agg['return']) - 1
        df_agg['cumulative_fee'] = df_agg['fee'].cumsum()
        df_agg['cumulative_return'] = (1 + df_agg['return_net']).cumprod()
        df_agg['possible_maximum_drawdown'] = get_possible_maximum_drawdown(df_agg['cumulative_return'])
        return df_agg
