import pandas as pd
class Alphas:
    def __init__(self):
        pass

    def simple_momentum_nday_rank(self, dict_df_klines: dict, n=1):
        '''
        weight = rank(close price change compared to n days ago)
        '''
        df_agg = pd.concat(
            [df_klines['close'].astype('float').pct_change(n).shift(1).rename(f'{symbol}_weight') for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_rank = df_agg.rank(axis=1)
        return df_rank

    def simple_regression_nday_rank(self, dict_df_klines: dict, n=1):
        '''
        weight = -rank(close price change compared to n days ago)
        '''
        df_agg = pd.concat(
            [-df_klines['close'].astype('float').pct_change(n).shift(1).rename(f'{symbol}_weight') for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_rank = df_agg.rank(axis=1)
        return df_rank

    def simple_momentum_nday(self, dict_df_klines: dict, n=1):
        '''
        weight = close price change compared to n days ago
        '''
        df_agg = pd.concat(
            [df_klines['close'].astype('float').pct_change(n).shift(1).rename(f'{symbol}_weight') for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        return df_agg

    def simple_regression_nday(self, dict_df_klines: dict, n=1):
        '''
        weight = -(close price change compared to n days ago)
        '''
        df_agg = pd.concat(
            [-df_klines['close'].astype('float').pct_change(n).shift(1).rename(f'{symbol}_weight') for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        return df_agg

    def simple_volume_nday_rank(self, dict_df_klines: dict, n=20):
        '''
        weight = rank(volume / nday volume mean)
        '''
        df_agg = pd.concat(
            [(df_klines['volume'].astype('float') / df_klines['volume'].astype('float').rolling(n).mean()).shift(1).rename(f'{symbol}_weight') for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_rank = df_agg.rank(axis=1)
        return df_rank

    def simple_volume_nday(self, dict_df_klines: dict, n=20):
        '''
        weight = volume / nday volume mean
        '''
        df_agg = pd.concat(
            [(df_klines['volume'].astype('float') / df_klines['volume'].astype('float').rolling(n).mean()).shift(1).rename(f'{symbol}_weight') for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        return df_agg