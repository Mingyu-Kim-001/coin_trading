import pandas as pd
from utils import *
class Alphas:
    def __init__(self):
        pass

    def close_momentum_nday_rank(self, dict_df_klines: dict, n=1):
        '''
        weight = rank(close price change compared to n days ago)
        '''
        df_agg = pd.concat(
            [df_klines['close'].astype('float').pct_change(n).shift(1).rename(f'{symbol}_weight') for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_rank = df_agg.rank(axis=1)
        df_neutralized_weight = neutralize_weight(df_rank)
        return df_neutralized_weight

    def close_regression_nday_rank(self, dict_df_klines: dict, n=1):
        '''
        weight = -rank(close price change compared to n days ago)
        '''
        df_agg = pd.concat(
            [-df_klines['close'].astype('float').pct_change(n).shift(1).rename(f'{symbol}_weight') for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_rank = df_agg.rank(axis=1)
        df_neutralized_weight = neutralize_weight(df_rank)
        return df_neutralized_weight

    def close_momentum_nday(self, dict_df_klines: dict, n=1):
        '''
        weight = close price change compared to n days ago
        '''
        df_agg = pd.concat(
            [df_klines['close'].astype('float').pct_change(n).shift(1).rename(f'{symbol}_weight') for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    def close_regression_nday(self, dict_df_klines: dict, n=1):
        '''
        weight = -(close price change compared to n days ago)
        '''
        df_agg = pd.concat(
            [-df_klines['close'].astype('float').pct_change(n).shift(1).rename(f'{symbol}_weight') for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    def simple_volume_nday_rank(self, dict_df_klines: dict, n=20):
        '''
        weight = rank(volume / nday volume mean)
        '''
        df_agg = pd.concat(
            [(df_klines['volume'].astype('float') / df_klines['volume'].astype('float').rolling(n).mean()).shift(1).rename(f'{symbol}_weight') for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_rank = df_agg.rank(axis=1)
        df_neutralized_weight = neutralize_weight(df_rank)
        return df_neutralized_weight

    def simple_volume_nday(self, dict_df_klines: dict, n=20):
        '''
        weight = volume / nday volume mean
        '''
        df_agg = pd.concat(
            [(df_klines['volume'].astype('float') / df_klines['volume'].astype('float').rolling(n).mean()).shift(1).rename(f'{symbol}_weight') for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    def correlation_open_close_nday(self, dict_df_klines:dict, n=10):
        '''
        weight = correlation(open, close, n)
        '''
        df_agg = pd.concat(
            [df_klines['open'].shift(1).rolling(n).corr(df_klines['close'].shift(1)).rename(f'{symbol}_weight') for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight