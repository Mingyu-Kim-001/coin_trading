import datetime
import pandas as pd
from alpha_collection import Alphas
import matplotlib.pyplot as plt
import os
from utils import *

FEE_RATE = 0.0002

def get_binance_klines_data_1d(symbol, start_datetime=datetime.datetime(2017, 1, 1, 9, 0, 0),
                               end_datetime=datetime.datetime(2022, 6, 1, 0, 0, 0), freq='1d', is_future=False):
    with open(f'./coin_backdata_hourly/{"f" + symbol if is_future else symbol}.csv', 'r') as f:
        df = pd.read_csv(f)
        df['timestamp'] = df['timestamp'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df_date_dummy = pd.DataFrame({'timestamp': pd.date_range(start_datetime, end_datetime, freq=freq)})
    df_extended = df_date_dummy.merge(df, on='timestamp', how='left').fillna(0)
    for column in ['open', 'high', 'low', 'close', 'volume']:
        df_extended[column] = df_extended[column].astype(float)
    if symbol == 'DOGEUSDT':
        df_extended = df_extended[df_extended['timestamp'] >= pd.to_datetime('2022-01-01')]  # remove volatile data
    return df_extended


def get_binance_klines_data_1h(symbol, start_datetime=datetime.datetime(2017, 1, 1, 9, 0, 0),
                               end_datetime=datetime.datetime(2022, 6, 1, 0, 0, 0), freq='1h', is_future=False):
    with open(f'./coin_backdata_hourly/{"f" + symbol if is_future else symbol}.csv', 'r') as f:
        df = pd.read_csv(f)
        df['timestamp'] = df['timestamp'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df_date_dummy = pd.DataFrame({'timestamp': pd.date_range(start_datetime, end_datetime, freq=freq)})
    df_extended = df_date_dummy.merge(df, on='timestamp', how='left').fillna(0)
    for column in ['open', 'high', 'low', 'close', 'volume']:
        df_extended[column] = df_extended[column].astype(float)
    if symbol == 'DOGEUSDT':
        df_extended = df_extended[df_extended['timestamp'] >= pd.to_datetime('2022-01-01')]  # remove volatile data
    return df_extended


def backtest_coin_strategy(df_neutralized_weight, dict_df_klines, symbols, stop_loss=-1,
                           leverage=1):
    dict_df_return, dict_df_trade_size = {}, {}
    df_timestamp = list(dict_df_klines.values())[0]
    df_agg = pd.DataFrame(df_timestamp, columns=['timestamp'], index=df_neutralized_weight.index)
    for symbol in symbols:
        df_neutralized_weight_symbol = df_neutralized_weight[symbol]
        pct_change = dict_df_klines[symbol]['open'].pct_change().shift(-1).fillna(0).replace([-np.inf, np.inf], 0)
        daily_maximum_drawdown = (dict_df_klines[symbol]['open'] - dict_df_klines[symbol]['low']) / \
                                 dict_df_klines[symbol]['open'] - 1
        daily_maximum_gain = (dict_df_klines[symbol]['high'] - dict_df_klines[symbol]['open']) / dict_df_klines[symbol][
            'open'] - 1
        # is_stop_loss = ((df_neutralized_weight_symbol < 0) & (daily_maximum_drawdown < stop_loss)) #\
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
    df_agg['fee'] = df_agg['trade_size'].mul(FEE_RATE)
    df_agg['return_net'] = (1 - df_agg['fee']) * (1 + df_agg['return']) - 1
    df_agg['cumulative_fee'] = df_agg['fee'].cumsum()
    df_agg['cumulative_return'] = (1 + df_agg['return_net']).cumprod()
    df_agg['possible_maximum_drawdown'] = get_possible_maximum_drawdown(df_agg['cumulative_return'])
    return df_agg

def save_backtest_result_figure(backtest_result, alpha_name, start_date, end_date, leverage, is_future):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(backtest_result['timestamp'], backtest_result['cumulative_return'], label='cumulative_return')
    ax.set_xlim([start_date, end_date])
    ax.set_yscale('log')
    dir_name = f'./figures/spot/{alpha_name}' if not is_future else f'./figures/future/{alpha_name}'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig.savefig(f'{dir_name}/{start_date}~{end_date}_leverage={leverage}.png')

def log_backtest_result(backtest_result, date_start, date_end, is_future, is_save_figure):
    final_return = round(backtest_result['cumulative_return'].iloc[-1], 2)
    possible_maximum_drawdown = round(backtest_result['possible_maximum_drawdown'].min(), 2)
    win_rate = round(sum(backtest_result['return'] > 0) / len(backtest_result.loc[lambda x: x['return'] != 0]), 4)
    print(date_start, '~', date_end, 'final_return', final_return, ', possible_maximum_drawdown', possible_maximum_drawdown, ', win_rate', win_rate)
    if is_save_figure:
        save_backtest_result_figure(backtest_result, alpha_name, date_start, date_end, leverage, is_future)


def backtest_for_alpha(symbols, df_weight, dict_df_klines, datetime_start, datetime_end,  leverage=1, is_future=False, is_save_figure=False):
    backtest_result = backtest_coin_strategy(df_weight, dict_df_klines, symbols, leverage=leverage, stop_loss=-1)
    log_backtest_result(backtest_result, datetime_start, datetime_end, is_future, is_save_figure=is_save_figure)


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
alpahs = Alphas()
# alpha_org_names = [alpha_name for alpha_name in alpahs.__dir__() if not alpha_name.startswith('_')]
shift = 0
alpha_org_names = ['close_position_in_nday_bollinger_band_median']
dict_alphas = {}
for alpha_name in alpha_org_names:
    # if
    # if alpha_name == 'close_position_in_nday_bollinger_band':
    #     for n in [4, 20]:
    #         dict_alphas[alpha_name + f'_{n}'] = (lambda name, n: lambda x: getattr(alpahs, name)(x, n))(alpha_name, n)
    if 'nday' in alpha_name:
        # for n in [3,4,5,6,7,8,10,15,20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140]:
        # for n in [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140]:
        for n in [100]:
        # for n in [60]:
        # for n in [2]:
        # for n in [20]:
            dict_alphas[alpha_name + f'_{n}'] = (lambda name, n, shift: lambda x: getattr(alpahs, name)(x, n, shift=shift))(alpha_name, n, 1)
            # if alpha_name == 'close_momentum_nday':
            # for weight_max in [0.5, 0.7, 0.9, 1, 1.5]:
            #     dict_alphas[alpha_name + f'_{n}_weight_max_{weight_max}'] = (lambda name, n, weight_max: lambda x: getattr(alpahs, name)(x, n, weight_max))(alpha_name, n, weight_max)

    else:
        dict_alphas[alpha_name] = getattr(alpahs, alpha_name)


dict_df_klines = {}
# start_date = datetime.date(2017, 8, 17)
# date_1 = datetime.date(2019, 1, 1)
# date_2 = datetime.date(2020, 1, 1)
# date_3 = datetime.date(2021, 1, 1)
# date_4 = datetime.date(2022, 1, 1)
# date_5 = datetime.date(2023, 1, 1)
# end_date = datetime.date(2023, 6, 10)
# date_interval = [[start_date, end_date], [start_date, date_1], [date_1, date_2], [date_2, date_3], [date_3, date_4], [date_4, date_5], [date_4, end_date], [date_5, end_date]]
data_freq = '1h'
trade_freq = '8h'
shift = 8
leverage = 5
symbols = ['BTCUSDT', 'XRPUSDT', 'ETHUSDT', 'DOGEUSDT', 'LTCUSDT', 'MATICUSDT', 'TRXUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT']#, 'BCHUSDT']# 'BNBUSDT']


def filter_weight(df_weight, df_trade_timestamp_idx, additional_timing_to_trade_idx):
    # df_weight.loc[lambda x:~x.index.isin(df_data_range_idx)] = 0
    # new_weight = pd.DataFrame(index=range(len(df_weight)), columns=df_weight.columns)
    # new_weight.loc[df_trade_timestamp_idx] = df_weight.loc[df_trade_timestamp_idx]
    # new_weight.loc[additional_timing_to_trade_idx] = df_weight.loc[additional_timing_to_trade_idx]
    df_weight_filtered = df_weight.loc[lambda x: (x.index.isin(df_trade_timestamp_idx)) | (x.index.isin(additional_timing_to_trade_idx))]
    return df_weight_filtered

def extend_weight(df_weight, df_data_range_idx):
    df_weight_extended = pd.DataFrame(index=df_data_range_idx, columns=df_weight.columns)
    df_weight_extended.loc[df_weight.index] = df_weight.loc[df_weight.index]
    df_weight_extended = df_weight_extended.ffill().fillna(0)
    df_weight_extended.loc[~df_weight_extended.index.isin(df_data_range_idx)] = 0
    return df_weight_extended


dict_df_klines_futures = {}
hour = 8
future_start_date = datetime.datetime(2019, 9, 8, hour, 0, 0)
# future_start_date = datetime.datetime(2020, 1, 1, 9, 0, 0)
date_1 = datetime.datetime(2021, 1, 1, hour, 0, 0)
date_2 = datetime.datetime(2022, 1, 1, hour, 0, 0)
date_3 = datetime.datetime(2023, 1, 1, hour, 0, 0)
date_4 = datetime.datetime(2023, 7, 1, hour, 0, 0)
future_end_date = datetime.datetime(2023, 7, 14, hour, 0, 0)
date_intervals = [[future_start_date, future_end_date], [future_start_date, date_1], [date_1, date_2], [date_2, date_3], [date_3, future_end_date], [date_2, future_end_date], [date_4, future_end_date]]
for symbol in symbols:
    dict_df_klines_futures[symbol] = get_binance_klines_data_1h(symbol, future_start_date, future_end_date, freq=data_freq, is_future=True)
for alpha_name, alpha in dict_alphas.items():
    print(alpha_name)
    df_weight, additional_timing_to_trade_idx = alpha(dict_df_klines_futures)
    for datetime_start, datetime_end in date_intervals:
        df_trade_timestamp_idx = list(dict_df_klines_futures.values())[0].loc[lambda x: x.timestamp.isin(pd.date_range(datetime_start, datetime_end, freq=trade_freq))].index # to restore index
        df_data_range_idx = list(dict_df_klines_futures.values())[0].loc[lambda x: x.timestamp.isin(pd.date_range(datetime_start, datetime_end, freq=data_freq))].index

        ###
        additional_timing_to_trade_idx = df_trade_timestamp_idx
        ###

        df_weight_filtered = filter_weight(df_weight, df_trade_timestamp_idx, additional_timing_to_trade_idx)
        df_weight_filtered = df_weight_filtered.shift(1)
        df_weight_extended = extend_weight(df_weight_filtered, df_data_range_idx)
        backtest_for_alpha(symbols, df_weight_extended, dict_df_klines_futures, datetime_start, datetime_end, leverage=leverage, is_future=True, is_save_figure=False)
    print()
print('-------------------')



