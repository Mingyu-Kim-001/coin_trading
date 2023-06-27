import datetime
import pandas as pd
from coin_trading_backtest import market_neutral_trading_backtest_binance
from alpha_collection import Alphas
import matplotlib.pyplot as plt
import os

def save_backtest_result_figure(backtest_result, alpha_name, start_date, end_date, leverage, is_future):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(backtest_result['date'], backtest_result['cumulative_return'], label='cumulative_return')
    ax.set_xlim([start_date, end_date])
    ax.set_yscale('log')
    dir_name = f'./figures/spot/{alpha_name}' if not is_future else f'./figures/future/{alpha_name}'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig.savefig(f'{dir_name}/{start_date}~{end_date}_leverage={leverage}.png')

def print_backtest_result(dict_alphas, dict_df_klines, df_date, date_interval, leverage=1, is_future=False, is_save_figure=False):
    for alpha_name, alpha in dict_alphas.items():
        df_weight = alpha(dict_df_klines)
        final_return, possible_maximum_drawdown, win_day_rate = [], [], []
        print(alpha_name)
        for date_interval_start, date_interval_end in date_interval:
            df_weight_interval = df_weight.copy()
            df_weight_interval.loc[
            (df_date < pd.to_datetime(date_interval_start)) | (df_date > pd.to_datetime(date_interval_end)), :] = 0
            backtest_result = backtest.backtest_coin_strategy(df_weight_interval, dict_df_klines, df_date, symbols,
                                                              leverage=leverage)
            final_return.append(round(backtest_result['cumulative_return'].iloc[-1], 2))
            possible_maximum_drawdown.append(round(backtest_result['possible_maximum_drawdown'].min(), 2))
            win_day_rate.append(
                round(sum(backtest_result['return'] > 0) / len(backtest_result.loc[lambda x: x['return'] != 0]), 4))
            print('final_return', final_return[-1], 'possible_maximum_drawdown', possible_maximum_drawdown[-1],
                  [f'{date_interval_start} ~ {date_interval_end}'], 'win_day_rate', win_day_rate[-1])
        start_date = min([date_int[0] for date_int in date_interval])
        end_date = max([date_int[1] for date_int in date_interval])
        if is_save_figure:
            backtest_result = backtest.backtest_coin_strategy(df_weight, dict_df_klines, df_date, symbols, leverage=leverage)
            save_backtest_result_figure(backtest_result, alpha_name, start_date, end_date, leverage, is_future)
        print()


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
backtest = market_neutral_trading_backtest_binance()
alpahs = Alphas()
# alpha_org_names = [alpha_name for alpha_name in alpahs.__dir__() if not alpha_name.startswith('_')]
alpha_org_names = ['close_in_nday_bollinger_band', 'bollinger_band_nday']
dict_alphas = {}
for alpha_name in alpha_org_names:
    if 'nday' in alpha_name:
        for n in [4, 20]:
        # for n in list(range(2, 10)) + [15, 20]:
            dict_alphas[alpha_name + f'_{n}'] = (lambda name, n: lambda x: getattr(alpahs, name)(x, n))(alpha_name, n)
            # if alpha_name == 'close_momentum_nday':
            # for weight_max in [0.5, 0.7, 0.9, 1, 1.5]:
            #     dict_alphas[alpha_name + f'_{n}_weight_max_{weight_max}'] = (lambda name, n, weight_max: lambda x: getattr(alpahs, name)(x, n, weight_max))(alpha_name, n, weight_max)
    else:
        dict_alphas[alpha_name] = getattr(alpahs, alpha_name)


dict_df_klines = {}
start_date = datetime.date(2017, 8, 17)
date_1 = datetime.date(2019, 1, 1)
date_2 = datetime.date(2020, 1, 1)
date_3 = datetime.date(2021, 1, 1)
date_4 = datetime.date(2022, 1, 1)
date_5 = datetime.date(2023, 1, 1)
end_date = datetime.date(2023, 6, 10)
date_interval = [[start_date, end_date], [start_date, date_1], [date_1, date_2], [date_2, date_3], [date_3, date_4], [date_4, date_5], [date_4, end_date], [date_5, end_date]]
print(f'spot {start_date} ~ {end_date}')
symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'DOGEUSDT', 'LTCUSDT', 'MATICUSDT', 'TRXUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT']#, 'BNBUSDT', 'DOTUSDT']
leverage = 4
for symbol in symbols:
    dict_df_klines[symbol] = backtest.get_binance_klines_data_1d(symbol, start_date, end_date)
df_date = list(dict_df_klines.values())[0]['date']
print_backtest_result(dict_alphas, dict_df_klines, df_date, date_interval, leverage=leverage, is_future=False, is_save_figure=False)



print('-------------------')
dict_df_klines_futures = {}
future_start_date = datetime.date(2019, 9, 8)
date_1 = datetime.date(2020, 1, 1)
date_2 = datetime.date(2021, 1, 1)
date_3 = datetime.date(2022, 1, 1)
date_4 = datetime.date(2023, 1, 1)
future_end_date = datetime.date(2023, 6, 10)
date_interval = [[future_start_date, future_end_date], [future_start_date, date_1], [date_1, date_2], [date_2, date_3], [date_3, date_4], [date_4, future_end_date], [date_3, future_end_date]]
for symbol in symbols:
    dict_df_klines_futures[symbol] = backtest.get_binance_klines_data_1d(symbol, future_start_date, future_end_date, is_future=True)
print(f'future {future_start_date} ~ {future_end_date}')
df_date = list(dict_df_klines_futures.values())[0]['date']
print_backtest_result(dict_alphas, dict_df_klines_futures, df_date, date_interval, leverage=leverage, is_future=True, is_save_figure=True)
print('-------------------')



