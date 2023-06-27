import datetime
import pandas as pd
from coin_trading_backtest import market_neutral_trading_backtest_binance
from alpha_collection import Alphas
import matplotlib.pyplot as plt

def save_backtest_result_figure(backtest_result, alpha_name, start_date, end_date, leverage):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(backtest_result['date'], backtest_result['cumulative_return'], label='cumulative_return')
    ax.set_xlim([start_date, end_date])
    ax.set_yscale('log')
    fig.savefig(f'./figures/{alpha_name}_{start_date}~{end_date}_leverage={leverage}.png')

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
backtest = market_neutral_trading_backtest_binance()
alpahs = Alphas()
# alpha_org_names = [alpha_name for alpha_name in alpahs.__dir__() if not alpha_name.startswith('_')]
alpha_org_names = ['bollinger_band_nday']
dict_alphas = {}
for alpha_name in alpha_org_names:
    if 'nday' in alpha_name:
        for n in [4, 20]:
        # for n in list(range(1, 5)):# + [10, 20, 50, 100, 200]:
            dict_alphas[alpha_name + f'_{n}'] = (lambda name, n: lambda x: getattr(alpahs, name)(x, n))(alpha_name, n)
            # if alpha_name == 'close_momentum_nday':
            # for weight_max in [0.5, 0.7, 0.9, 1, 1.5]:
            #     dict_alphas[alpha_name + f'_{n}_weight_max_{weight_max}'] = (lambda name, n, weight_max: lambda x: getattr(alpahs, name)(x, n, weight_max))(alpha_name, n, weight_max)
    else:
        dict_alphas[alpha_name] = getattr(alpahs, alpha_name)


dict_df_klines = {}
start_date = datetime.date(2017, 8, 17)
past_recent_split_date = datetime.date(2023, 1, 1)
end_date = datetime.date(2023, 6, 10)
print(f'spot {start_date} ~ {end_date}')
symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'DOGEUSDT', 'LTCUSDT', 'MATICUSDT', 'TRXUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT']#, 'BNBUSDT', 'DOTUSDT']
leverage = 5 * 0.9
# symbols = ['BTCUSDT', 'ETHUSDT']

for symbol in symbols:
    dict_df_klines[symbol] = backtest.get_binance_klines_data_1d(symbol, start_date, end_date)
df_date = list(dict_df_klines.values())[0]['date']
for alpha_name, alpha in dict_alphas.items():
    df_weight = alpha(dict_df_klines)
    df_weight_past = df_weight.copy()
    df_weight_recent = df_weight.copy()
    df_weight_past.loc[df_date >= pd.to_datetime(past_recent_split_date), :] = 0
    df_weight_recent.loc[df_date < pd.to_datetime(past_recent_split_date), :] = 0
    final_return, possible_maximum_drawdown, win_day_rate = [] , [], []
    for df_weight, is_past in zip([df_weight_past, df_weight_recent], [True, False]):
        backtest_result = backtest.backtest_coin_strategy(df_weight, dict_df_klines, df_date, symbols, leverage=leverage)
        final_return.append(round(backtest_result['cumulative_return'].iloc[-1], 2))
        possible_maximum_drawdown.append(round(backtest_result['possible_maximum_drawdown'].min(), 2))
        win_day_rate.append(round(sum(backtest_result['return'] > 0) / len(backtest_result.loc[lambda x: x['return'] != 0]), 4))
    print(alpha_name, 'final return', final_return, 'possible_maximum_drawdown', possible_maximum_drawdown, [f'{start_date} ~ {past_recent_split_date}', f'{past_recent_split_date} ~ {end_date}'], 'win_day_rate', win_day_rate)


print('-------------------')
dict_df_klines_futures = {}
future_start_date = datetime.date(2019, 9, 8)
future_past_recent_split_date = datetime.date(2022, 1, 1)
future_end_date = datetime.date(2023, 6, 10)
for symbol in symbols:
    dict_df_klines_futures[symbol] = backtest.get_binance_klines_data_1d(symbol, future_start_date, future_end_date, is_future=True)
print(f'future {future_start_date} ~ {future_end_date}')
df_date = list(dict_df_klines_futures.values())[0]['date']
df_weights = {}
for alpha_name, alpha in dict_alphas.items():
    df_weights[alpha_name] = alpha(dict_df_klines_futures)
    df_weight_past = df_weights[alpha_name].copy()
    df_weight_recent = df_weights[alpha_name].copy()
    df_weight_past.loc[df_date >= pd.to_datetime(future_past_recent_split_date), :] = 0
    df_weight_recent.loc[df_date < pd.to_datetime(future_past_recent_split_date), :] = 0
    final_return, possible_maximum_drawdown, win_day_rate = [] , [], []
    for df_weight, is_past in zip([df_weight_past, df_weight_recent], [True, False]):
        backtest_result = backtest.backtest_coin_strategy(df_weight, dict_df_klines_futures, df_date, symbols, leverage=leverage)
        final_return.append(round(backtest_result['cumulative_return'].iloc[-1], 2))
        possible_maximum_drawdown.append(round(backtest_result['possible_maximum_drawdown'].min(), 2))
        win_day_rate.append(
            round(sum(backtest_result['return'] > 0) / len(backtest_result.loc[lambda x: x['return'] != 0]), 4))
        start_date = future_start_date if is_past else future_past_recent_split_date
        end_date = future_past_recent_split_date if is_past else future_end_date
        save_backtest_result_figure(backtest_result, alpha_name, start_date, end_date, leverage=leverage)
    print(alpha_name,'whole day final return', round(final_return[0] * final_return[1], 2), 'final return', final_return, 'possible_maximum_drawdown', possible_maximum_drawdown, [f'{future_start_date} ~ {future_past_recent_split_date}', f'{future_past_recent_split_date} ~ {future_end_date}'], 'win_day_rate', win_day_rate)

print('-------------------')



