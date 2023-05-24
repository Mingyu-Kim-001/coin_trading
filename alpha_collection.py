import pandas as pd
class Alphas:
    def __init__(self):
        self.alpha_list = ['simple_momentum_strategy', 'simple_regression_strategy']

    def simple_momentum_strategy(self, dict_df_klines: dict):
        df_agg = pd.concat(
            [df_klines['close'].astype('float').pct_change().shift(1).rename(f'{symbol}_weight') for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_rank = df_agg.rank(axis=1)
        return df_rank

    def simple_regression_strategy(self, dict_df_klines: dict):
        df_agg = pd.concat(
            [-df_klines['close'].astype('float').pct_change().shift(1).rename(f'{symbol}_weight') for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_rank = df_agg.rank(axis=1)
        return df_rank

