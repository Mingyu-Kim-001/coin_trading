import pandas as pd
class Alphas:
    def __init__(self):
        pass

    def simple_momentum_nday_strategy(self, dict_df_klines: dict, n=3):
        df_agg = pd.concat(
            [df_klines['close'].astype('float').pct_change().shift(n).rename(f'{symbol}_weight') for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_rank = df_agg.rank(axis=1)
        return df_rank

    def simple_regression_nday_strategy(self, dict_df_klines: dict, n=3):
        df_agg = pd.concat(
            [-df_klines['close'].astype('float').pct_change().shift(n).rename(f'{symbol}_weight') for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_rank = df_agg.rank(axis=1)
        return df_rank