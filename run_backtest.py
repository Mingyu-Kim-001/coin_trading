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
    return df_extended

def get_binance_klines_data_1m(symbol, start_datetime=datetime.datetime(2017, 1, 1, 9, 0, 0),
                               end_datetime=datetime.datetime(2022, 6, 1, 0, 0, 0), freq='1m', is_future=False):
    with open(f'./coin_backdata_hourly/{"f" + symbol if is_future else symbol}.csv', 'r') as f:
        df = pd.read_csv(f)
        df['timestamp'] = df['timestamp'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df_date_dummy = pd.DataFrame({'timestamp': pd.date_range(start_datetime, end_datetime, freq=freq)})
    df_extended = df_date_dummy.merge(df, on='timestamp', how='left').fillna(0)
    for column in ['open', 'high', 'low', 'close', 'volume']:
        df_extended[column] = df_extended[column].astype(float)
    return df_extended


def backtest_coin_strategy(df_neutralized_weight, dict_df_klines, symbols, stop_loss=-1,
                           leverage=1):
    dict_df_return, dict_df_trade_size = {}, {}
    df_timestamp = list(dict_df_klines.values())[0]
    df_agg = pd.DataFrame(df_timestamp, columns=['timestamp'], index=df_neutralized_weight.index)
    for symbol in symbols:
        df_neutralized_weight_symbol = df_neutralized_weight[symbol]
        df_klines = dict_df_klines[symbol].loc[lambda x:x.index.isin(df_neutralized_weight_symbol.index)]
        pct_change = df_klines['open'].pct_change().shift(-1).fillna(0).replace([-np.inf, np.inf], 0)
        daily_maximum_drawdown = (dict_df_klines[symbol]['low'] - dict_df_klines[symbol]['open']) / \
                                 dict_df_klines[symbol]['open']
        daily_maximum_gain = (dict_df_klines[symbol]['high'] - dict_df_klines[symbol]['open']) / dict_df_klines[symbol][
            'open']
        is_stop_loss = ((df_neutralized_weight_symbol < 0) & (daily_maximum_drawdown < stop_loss)) #\
        #                | ((df_neutralized_weight_symbol > 0) & (daily_maximum_gain > -stop_loss))
        dict_df_return[symbol] = pd.Series(np.where(is_stop_loss,
                                                    -abs(stop_loss * df_neutralized_weight_symbol),
                                                    pct_change * df_neutralized_weight_symbol
                                                    ), index=df_neutralized_weight.index)
        # dict_df_return[symbol] = pct_change * df_neutralized_weight_symbol
        df_neutralized_weight_symbol_lag = df_neutralized_weight_symbol.shift(1)

        dict_df_trade_size[symbol] = pd.Series(
            np.where(is_stop_loss,
                     np.where(is_stop_loss.shift(1),
                              2 * abs(df_neutralized_weight_symbol),
                              abs(df_neutralized_weight_symbol) + abs(
                                  df_neutralized_weight_symbol_lag - df_neutralized_weight_symbol)),
                     np.where(is_stop_loss.shift(1),
                              abs(df_neutralized_weight_symbol),
                              abs(df_neutralized_weight_symbol - df_neutralized_weight_symbol_lag))), index=df_neutralized_weight.index)
        # dict_df_trade_size[symbol] = abs(df_neutralized_weight_symbol - df_neutralized_weight_symbol_lag)

    df_agg['return'] = (pd.DataFrame(dict_df_return, index=df_neutralized_weight.index).sum(axis=1) * leverage).clip(-1, float("inf"))
    df_agg['trade_size'] = pd.DataFrame(dict_df_trade_size, index=df_neutralized_weight.index).sum(axis=1) * leverage
    df_agg['fee'] = df_agg['trade_size'].mul(FEE_RATE)
    df_agg['return_net'] = (1 - df_agg['fee']) * (1 + df_agg['return']) - 1
    df_agg['cumulative_fee'] = df_agg['fee'].cumsum()
    df_agg['cumulative_return'] = (1 + df_agg['return_net']).cumprod()
    df_agg['possible_maximum_drawdown'] = get_possible_maximum_drawdown(df_agg['cumulative_return'])
    df_agg['one_shot_maximum_drawdown'] = get_maximum_drawdown_one_shot(df_agg['cumulative_return'])
    return df_agg

def save_backtest_result_figure(backtest_result, alpha_name, start_date, end_date, leverage, is_future):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(backtest_result['timestamp'], backtest_result['cumulative_return'], label='cumulative_return')
    ax.set_xlim([start_date, end_date])
    ax.set_yscale('log')
    n = alpha_name.rsplit('_', 1)[1] if 'nday' in alpha_name else None
    alpha_without_n = alpha_name.rsplit('_', 1)[0] + f'/n={n}' if 'nday' in alpha_name else alpha_name
    dir_name = f'./figures/spot/{alpha_without_n}' if not is_future else f'./figures/future/{alpha_without_n}'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig.savefig(f'{dir_name}/{start_date}~{end_date}_leverage={leverage}.png')
    plt.close(fig)

def log_backtest_result(alpha_name, backtest_result, date_start, date_end, is_future, is_save_figure):
    final_return = round(backtest_result['cumulative_return'].iloc[-1], 2)
    possible_maximum_drawdown = round(backtest_result['possible_maximum_drawdown'].min(), 2)
    one_shot_maximum_drawdown = round(backtest_result['one_shot_maximum_drawdown'].min(), 2)
    win_rate = round(sum(backtest_result['return'] > 0) / len(backtest_result.loc[lambda x: x['return'] != 0]), 4)
    sharp_ratio = round(backtest_result['return'].mean() / backtest_result['return'].std(), 2)
    print(date_start, '~', date_end, 'final_return', final_return, ', possible_maximum_drawdown', possible_maximum_drawdown, ', one_shot_maximum_drawdown', one_shot_maximum_drawdown, ', win_rate', win_rate, ', sharp_ratio', sharp_ratio)
    if is_save_figure:
        save_backtest_result_figure(backtest_result, alpha_name, date_start, date_end, leverage, is_future)


def backtest_for_alpha(alpha_name, symbols, df_weight, dict_df_klines, datetime_start, datetime_end,  leverage=1, stop_loss=-1, is_future=False, is_save_figure=False):
    backtest_result = backtest_coin_strategy(df_weight, dict_df_klines, symbols, leverage=leverage, stop_loss=stop_loss)
    log_backtest_result(alpha_name, backtest_result, datetime_start, datetime_end, is_future, is_save_figure=is_save_figure)

def fit_weight_and_klines(df_weight, dict_df_klines, df_trade_timestamp_idx, additional_timing_to_trade_idx):
    if additional_timing_to_trade_idx is not None:
        additional_timing_to_trade_idx_fit = [idx for idx in additional_timing_to_trade_idx if
                                              idx > df_trade_timestamp_idx[0] and idx < df_trade_timestamp_idx[-1]]
        df_weight_filtered = df_weight.loc[lambda x:(x.index.isin(df_trade_timestamp_idx)) | (x.index.isin(additional_timing_to_trade_idx_fit))]
    else:
        df_weight_filtered = df_weight.loc[lambda x: x.index.isin(df_trade_timestamp_idx)]
    weight_index = df_weight_filtered.index
    lag_weight_index = [weight_index[i+1] if i < len(weight_index) - 1 else max(weight_index) + 1 for i in range(len(weight_index))]
    dict_df_klines_fit = {}
    for symbol, df_klines in dict_df_klines.items():
        df_tmp = pd.DataFrame(index=df_klines.index)
        df_tmp.loc[weight_index, 'new_idx'] = weight_index
        df_tmp.ffill(inplace=True)
        df_klines['new_idx'] = df_tmp['new_idx']
        df_klines['original_idx'] = df_klines.index
        df_klines_fit = query_on_pandas_df("""--sql
            WITH rn_added
                     AS (SELECT new_idx, timestamp, open, high, low, close, volume, new_idx, original_idx, 
                                ROW_NUMBER() OVER (PARTITION BY new_idx ORDER BY original_idx)      rn_asc,
                                ROW_NUMBER() OVER (PARTITION BY new_idx ORDER BY original_idx DESC) rn_desc
                         FROM df_klines)
            SELECT new_idx,                
                   MAX(IF(rn_asc = 1, timestamp, NULL)) timestamp_tmp, --column name issue
                   MAX(IF(rn_asc = 1, open, NULL))      open,
                   MAX(high)                            high,
                   MIN(low)                             low,
                   MAX(IF(rn_desc = 1, close, NULL))    close_tmp, --column name issue
                   SUM(volume)                          volume,
            FROM rn_added
            WHERE new_idx IS NOT NULL
            GROUP BY 1
            ORDER BY 1;
        """)
        df_klines_fit['new_idx'] = df_klines_fit['new_idx'].astype(int)
        df_klines_fit = df_klines_fit.rename(columns={'timestamp_tmp':'timestamp', 'close_tmp':'close'})
        df_klines_fit = df_klines_fit.set_index('new_idx')
        dict_df_klines_fit[symbol] = df_klines_fit
    return df_weight_filtered, dict_df_klines_fit

def run_single_alpha_consequently(dict_df_klines, dict_alphas, symbols, date_intervals, leverage, stop_loss, is_future, is_save_figure):
    for alpha_name, alpha in dict_alphas.items():
        print(alpha_name)
        df_weight, additional_timing_to_trade_idx = alpha(dict_df_klines)
        for datetime_start, datetime_end in date_intervals:
            df_trade_timestamp_idx = list(dict_df_klines.values())[0].loc[lambda x: x.timestamp.isin(pd.date_range(datetime_start, datetime_end, freq=trade_freq))].index # to restore index
            df_data_range_idx = list(dict_df_klines.values())[0].loc[lambda x: x.timestamp.isin(pd.date_range(datetime_start, datetime_end, freq=data_freq))].index
            df_weight_filtered, dict_df_klines_fit = fit_weight_and_klines(df_weight, dict_df_klines, df_trade_timestamp_idx, additional_timing_to_trade_idx)
            backtest_for_alpha(alpha_name, symbols, df_weight_filtered, dict_df_klines_fit, datetime_start, datetime_end, leverage=leverage, stop_loss=stop_loss, is_future=is_future, is_save_figure=is_save_figure)
        print()

# def run_combine_alphas()

if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    alpahs = Alphas()
    # alpha_org_names = [alpha_name for alpha_name in alpahs.__dir__() if not alpha_name.startswith('_')]
    shift = 1
    alpha_org_names = ['simple_rsi', 'close_position_in_nday_bollinger_band_median']
    alpha_org_names = ['control_chart_rule1']
    dict_alphas = {}
    for alpha_name in alpha_org_names:
        # if
        # if alpha_name == 'close_position_in_nday_bollinger_band':
        #     for n in [4, 20]:
        #         dict_alphas[alpha_name + f'_{n}'] = (lambda name, n: lambda x: getattr(alpahs, name)(x, n))(alpha_name, n)
        if 'nday' in alpha_name:
            for n in [3,4,5,6,7,8,10,15,20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140]:
            # for n in [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200]:
            # for n in [95, 100, 105, 110, 115, 120, 125, 130, 135]:
                dict_alphas[alpha_name + f'_{n}'] = (lambda name, n, shift: lambda x: getattr(alpahs, name)(x, n, shift=shift))(alpha_name, n, shift)
                # if alpha_name == 'close_momentum_nday':
                # for weight_max in [0.5, 0.7, 0.9, 1, 1.5]:
                #     dict_alphas[alpha_name + f'_{n}_weight_max_{weight_max}'] = (lambda name, n, weight_max: lambda x: getattr(alpahs, name)(x, n, weight_max))(alpha_name, n, weight_max)

        else:
            dict_alphas[alpha_name] = getattr(alpahs, alpha_name)

    dict_df_klines = {}
    data_freq = '1h'
    trade_freq = '8h'
    leverage = 1
    symbols = ['BTCUSDT', 'XRPUSDT', 'ETHUSDT', 'DOGEUSDT', 'LTCUSDT', 'MATICUSDT', 'TRXUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT'] #, 'XLMUSDT']#, 'BNBUSDT', 'BCHUSDT']:
    dict_df_klines_futures = {}
    hour = 8
    future_start_date = datetime.datetime(2019, 9, 10, hour, 0, 0)
    # future_start_date = datetime.datetime(2020, 1, 1, 9, 0, 0)
    date_1 = datetime.datetime(2021, 1, 1, hour, 0, 0)
    date_2 = datetime.datetime(2022, 1, 1, hour, 0, 0)
    date_3 = datetime.datetime(2023, 1, 1, hour, 0, 0)
    date_4 = datetime.datetime(2023, 9, 1, hour, 0, 0)
    future_end_date = datetime.datetime(2023, 9, 30, hour, 0, 0)
    date_intervals = [[future_start_date, future_end_date], [future_start_date, date_1], [date_1, date_2], [date_2, date_3], [date_3, future_end_date], [date_2, future_end_date], [date_4, future_end_date]]
    for symbol in symbols:
        dict_df_klines_futures[symbol] = get_binance_klines_data_1h(symbol, future_start_date, future_end_date, freq=data_freq, is_future=True)
    run_single_alpha_consequently(dict_df_klines_futures, dict_alphas, symbols, date_intervals, leverage, stop_loss=-1, is_future=True, is_save_figure=False)



