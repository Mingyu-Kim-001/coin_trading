import pandas as pd
from utils import *
from coin_trading_backtest import market_neutral_trading_backtest_binance
from numba import jit
class Alphas:
    def __init__(self):
        pass

    def hold_bitcoin(self, dict_df_klines: dict):
        '''
        weight = 1 for bitcoin
        '''
        df_agg = pd.concat([pd.Series([1] * len(df_klines) if symbol == 'BTCUSDT' else [0] * len(df_klines), index=df_klines.index, name=symbol).rename(symbol) for symbol, df_klines in dict_df_klines.items()], axis=1)
        df_agg = df_agg.fillna(0)
        return df_agg

    def close_momentum_nday_rank(self, dict_df_klines: dict, n=1):
        '''
        weight = rank(close price change compared to n days ago)
        '''
        df_agg = pd.concat(
            [df_klines['close'].astype('float').pct_change(n).shift(1).rename(symbol) for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_rank = df_agg.rank(axis=1)
        df_neutralized_weight = neutralize_weight(df_rank)
        return df_neutralized_weight

    def close_regression_nday_rank(self, dict_df_klines: dict, n=1):
        '''
        weight = -rank(close price change compared to n days ago)
        '''
        df_agg = pd.concat(
            [-df_klines['close'].astype('float').pct_change(n).shift(1).rename(symbol) for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_rank = df_agg.rank(axis=1)
        df_neutralized_weight = neutralize_weight(df_rank)
        return df_neutralized_weight

    def close_momentum_nday(self, dict_df_klines: dict, n=1, weight_max=None, shift=1):
        '''
        weight = close price change compared to n days ago
        '''
        df_agg = pd.concat(
            [df_klines['close'].astype('float').pct_change(n).shift(shift).rename(symbol) for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        if weight_max is not None:
            df_agg = df_agg.clip(-weight_max, weight_max)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    def close_regression_nday(self, dict_df_klines: dict, n=1):
        '''
        weight = -(close price change compared to n days ago)
        '''
        df_agg = pd.concat(
            [-df_klines['close'].astype('float').pct_change(n).shift(1).rename(symbol) for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    def simple_volume_nday_rank(self, dict_df_klines: dict, n=20):
        '''
        weight = rank(volume / nday volume mean)
        '''
        df_agg = pd.concat(
            [(df_klines['volume'].astype('float') / df_klines['volume'].astype('float').rolling(n).mean()).shift(1).rename(symbol) for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_rank = df_agg.rank(axis=1)
        df_neutralized_weight = neutralize_weight(df_rank)
        return df_neutralized_weight

    def simple_volume_nday(self, dict_df_klines: dict, n=20):
        '''
        weight = volume / nday volume mean
        '''
        df_agg = pd.concat(
            [(df_klines['volume'].astype('float') / df_klines['volume'].astype('float').rolling(n).mean()).shift(1).rename(symbol) for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    def correlation_open_close_nday(self, dict_df_klines:dict, n=10):
        '''
        weight = correlation(open, close, n)
        '''
        df_agg = pd.concat(
            [df_klines['open'].shift(1).rolling(n).corr(df_klines['close'].shift(1)).rename(symbol) for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    def close_position_in_high_and_low(self, dict_df_klines:dict):
        '''
        weight = close position in high and low
        '''
        df_agg = pd.concat(
            [((df_klines['close'].astype('float') - df_klines['low'].astype('float')) / (df_klines['high'].astype('float') - df_klines['low'].astype('float'))).shift(1).rename(symbol) for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    def close_position_in_moving_average_nday(self, dict_df_klines:dict, n=10):
        '''
        weight = close position in moving average
        '''
        df_agg = pd.concat(
            [((df_klines['close'].astype('float') - df_klines['close'].astype('float').rolling(n).mean()) / df_klines['close'].astype('float').rolling(n).mean()).shift(1).rename(symbol) for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    def close_position_in_nday_bollinger_band(self, dict_df_klines:dict, n=20, weight_max=None, shift=1):
        '''
        weight = close position in bollinger band
        '''
        df_agg = pd.concat(
            [((df_klines['close'].astype('float') - df_klines['close'].astype('float').rolling(n).mean().shift(1)) / df_klines['close'].astype('float').rolling(n).std().shift(1)).shift(shift).rename(symbol) for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        if weight_max is not None:
            df_agg = df_agg.clip(-weight_max, weight_max)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    def close_position_in_nday_bollinger_band_median(self, dict_df_klines:dict, n=20, weight_max=None, shift=1):
        '''
        weight = close position in bollinger band
        '''
        df_agg = pd.concat(
            [((df_klines['close'].astype('float') - df_klines['close'].astype('float').rolling(n).median().shift(1)) / df_klines['close'].astype('float').rolling(n).std().shift(1)).shift(shift).rename(symbol) for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        if weight_max is not None:
            df_agg = df_agg.clip(-weight_max, weight_max)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    def close_position_in_nday_bollinger_band_std(self, dict_df_klines:dict, n=20, weight_max=None, shift=1):
        '''
        weight = close position in bollinger band * std
        '''
        df_agg = pd.concat(
            [((df_klines['close'].astype('float') - df_klines['close'].astype('float').rolling(n).mean().shift(1)) / df_klines['close'].astype('float').rolling(n).std().shift(1)).shift(shift).rename(symbol) for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_std = pd.concat([df_klines['close'].astype('float').rolling(n).std().shift(1).rename(symbol) for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_weight = neutralize_weight(df_agg) * df_std
        df_neutralized_weight = neutralize_weight(df_weight)
        return df_neutralized_weight

    def close_position_in_nday_bollinger_band_square(self, dict_df_klines:dict, n=20, weight_max=None, shift=1):
        '''
        weight = signed_square(close position in bollinger band)
        '''
        df_agg = pd.concat(
            [((df_klines['close'].astype('float') - df_klines['close'].astype('float').rolling(n).mean().shift(1)) / df_klines['close'].astype('float').rolling(n).std().shift(1)).shift(shift).rename(symbol) for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        #square, with consistent
        df_agg = df_agg * df_agg * np.sign(df_agg)
        if weight_max is not None:
            df_agg = df_agg.clip(-weight_max, weight_max)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    def close_position_in_nday_bollinger_band_rank(self, dict_df_klines:dict, n=20, weight_max=None, shift=1):
        '''
        weight = rank(close position in bollinger band)
        '''
        df_agg = pd.concat(
            [((df_klines['close'].astype('float') - df_klines['close'].astype('float').rolling(n).mean().shift(1)) / df_klines['close'].astype('float').rolling(n).std().shift(1)).shift(shift).rename(symbol) for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        df_rank = df_agg.rank(axis=1)
        df_neutralized_weight = neutralize_weight(df_rank)
        return df_neutralized_weight

    def high_in_nday_bollinger_band(self, dict_df_klines:dict, n=20, weight_max=None, shift=1):
        '''
        weight = high position in bollinger band
        '''
        df_agg = pd.concat(
            [((df_klines['high'].astype('float') - df_klines['close'].astype('float').rolling(n).mean().shift(1)) / df_klines['close'].astype('float').rolling(n).std().shift(1)).shift(shift).rename(symbol) for symbol, df_klines
             in dict_df_klines.items()], axis=1)
        if weight_max is not None:
            df_agg = df_agg.clip(-weight_max, weight_max)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    def bollinger_band_nday(self, dict_df_klines:dict, n=20, weight_max=None, shift=1):
        '''
        weight = bollinger band
        '''
        df_agg_list = []
        for symbol, df_klines in dict_df_klines.items():
            bollinger_band_std = df_klines['close'].astype('float').rolling(n).std()
            nday_moving_average = df_klines['close'].astype('float').rolling(n).mean()
            bollinger_band_upper = nday_moving_average + 2 * bollinger_band_std
            bollinger_band_lower = nday_moving_average - 2 * bollinger_band_std
            close_in_bollinger_band = (df_klines['close'].astype('float') - bollinger_band_lower) / bollinger_band_std
            df_agg_symbol = np.where((df_klines['close'] > bollinger_band_upper) | (df_klines['close'] < bollinger_band_lower), -1, 1) * close_in_bollinger_band
            df_agg_symbol = df_agg_symbol.shift(shift).rename(symbol)
            df_agg_list.append(df_agg_symbol)
        df_agg = pd.concat(df_agg_list, axis=1)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    def alpha_1_nday(self, dict_df_klines, n=20, shift=1):
        '''
        weight = (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, n) : close), 2.), 5)) - 0.5)
        '''
        df_agg_list = []
        for symbol, df_klines in dict_df_klines.items():
            df_klines['returns'] = df_klines['close'].astype('float').pct_change(1)
            df_klines['stddev'] = df_klines['returns'].rolling(n).std()
            df_agg_symbol = df_klines.apply(lambda x: x['stddev'] if x['returns'] < 0 else x['close'], axis=1)
            df_agg_symbol = df_agg_symbol.pow(2)
            df_agg_symbol = df_agg_symbol.rolling(5).apply(lambda x: np.argmax(x), raw=True)
            df_agg_symbol = df_agg_symbol.rank()
            df_agg_symbol = df_agg_symbol - 0.5
            df_agg_symbol = df_agg_symbol.shift(shift).rename(symbol)
            df_agg_list.append(df_agg_symbol)
        df_agg = pd.concat(df_agg_list, axis=1)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    def alpha_2_nday(self, dict_df_klines, n=6, shift=1):
        '''
        weight = (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
        '''
        df_agg_list = []
        for symbol, df_klines in dict_df_klines.items():
            df_klines['volume'] = df_klines['volume'].astype('float')
            df_klines['returns'] = df_klines['close'].astype('float').pct_change(1)
            df_klines['log_volume'] = np.log(df_klines['volume'])
            rank_delta_log_volume = df_klines['log_volume'].diff(2).rank()
            correlation = rank_delta_log_volume.rolling(n).corr(df_klines['returns'].rank())
            df_agg_symbol = -correlation.shift(shift).rename(symbol)
            df_agg_list.append(df_agg_symbol)
        df_agg = pd.concat(df_agg_list, axis=1)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    def alpha_3_nday(self, dict_df_klines, n=10, shift=1):
        '''
        weight = (-1 * correlation(rank(open), rank(volume), 10))
        '''
        df_agg_list = []
        for symbol, df_klines in dict_df_klines.items():
            df_klines['volume'] = df_klines['volume'].astype('float')
            df_agg_symbol = df_klines['open'].rank().rolling(n).corr(df_klines['volume'].rank())
            df_agg_symbol = -df_agg_symbol.shift(shift).rename(symbol)
            df_agg_list.append(df_agg_symbol)
        df_agg = pd.concat(df_agg_list, axis=1)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    def alpha_4_nday(self, dict_df_klines, n=9, shift=1):
        '''
        weight = (-1 * Ts_Rank(rank(low), 9))
        '''
        df_agg_list = []
        for symbol, df_klines in dict_df_klines.items():
            df_agg_symbol = df_klines['low'].rank().rolling(n).apply(lambda x: pd.Series(x).rank().iloc[-1])
            df_agg_symbol = -df_agg_symbol.shift(shift).rename(symbol)
            df_agg_list.append(df_agg_symbol)
        df_agg = pd.concat(df_agg_list, axis=1)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    # def alpha_5_nday(self, dict_df_klines, n=10, shift=1):
    #     '''
    #     weight = (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    #     '''
    #     df_agg_list = []
    #     for symbol, df_klines in dict_df_klines.items():
    #         df_klines['vwap'] = (df_klines['high'].astype('float') + df_klines['low'].astype('float') + df_klines['close'].astype('float')) / 3
    #         df_klines['vwap'] = df_klines.rolling(n).apply(lambda x: (x['high'].astype('float') + x['low'].astype('float') + x['close'].astype('float')) / 3, raw=True)
    #         df_agg_symbol = df_klines['open'].astype('float') - df_klines['vwap']


    def alpha_6_nday(self, dict_df_klines, n=10, shift=1):
        '''
        weight = (-1 * correlation(open, volume, 10))
        '''
        df_agg_list = []
        for symbol, df_klines in dict_df_klines.items():
            df_agg_symbol = df_klines['open'].astype('float').rolling(n).corr(df_klines['volume'].astype('float'))
            df_agg_symbol = -df_agg_symbol.shift(shift).rename(symbol)
            df_agg_list.append(df_agg_symbol)
        df_agg = pd.concat(df_agg_list, axis=1)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

    def alpha_7(self, dict_df_klines, shift=1):
        '''
        weight =  (adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : -1
        '''
        df_agg_list = []
        for symbol, df_klines in dict_df_klines.items():
            adv20 = df_klines['volume'].astype('float').rolling(20).mean()
            A = df_klines['close'].astype('float').diff(7).abs().rolling(60).apply(lambda x: pd.Series(x).rank().iloc[-1]) * np.sign(df_klines['close'].astype('float').diff(7))
            B = -1
            df_agg_symbol = pd.Series(np.where(adv20 < df_klines['volume'].astype('float'), A, B), index=df_klines.index)
            df_agg_symbol = df_agg_symbol.shift(shift).rename(symbol)
            df_agg_list.append(df_agg_symbol)
        df_agg = pd.concat(df_agg_list, axis=1)
        df_neutralized_weight = neutralize_weight(df_agg)
        return df_neutralized_weight

if __name__ == '__main__':
    symbols = ['BTCUSDT', 'ETHUSDT']
    start_date = '2022-01-01'
    end_date = '2022-06-30'
    alpha_name = 'alpha_7'
    backtest = market_neutral_trading_backtest_binance()
    dict_df_klines = {}
    for symbol in symbols:
        dict_df_klines[symbol] = backtest.get_binance_klines_data_1d(symbol, start_date, end_date)
    alpahs = Alphas()
    alpha = getattr(alpahs, alpha_name)
    df_weight = alpha(dict_df_klines)
    print()