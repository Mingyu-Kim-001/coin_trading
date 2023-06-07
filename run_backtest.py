import datetime
import pandas as pd
from coin_trading_backtest import market_neutral_trading_backtest_binance
from alpha_collection import Alphas

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
backtest = market_neutral_trading_backtest_binance()
alpahs = Alphas()
alpha_org_names = [alpha_name for alpha_name in alpahs.__dir__() if not alpha_name.startswith('_')]
dict_alphas = {}
for alpha_name in alpha_org_names:
    if 'nday' in alpha_name:
        for n in list(range(1, 5)) + [10, 20, 50, 100, 200]:
            dict_alphas[alpha_name + f'_{n}'] = (lambda name, n: lambda x: getattr(alpahs, name)(x, n))(alpha_name, n)
    else:
        dict_alphas[alpha_name] = getattr(alpahs, alpha_name)


dict_df_klines = {}
start_date = datetime.date(2017, 8, 17)
end_date = datetime.date(2023, 5, 1)
print(f'spot {start_date} ~ {end_date}')
symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'DOGEUSDT', 'LTCUSDT', 'MATICUSDT', 'TRXUSDT', 'ADAUSDT', 'SOLUSDT']
# symbols = ['BTCUSDT', 'ETHUSDT']

for symbol in symbols:
    dict_df_klines[symbol] = backtest.get_binance_klines_data_1d(symbol, start_date, end_date)
df_date = list(dict_df_klines.values())[0]['date']
past_recent_split_date = datetime.date(2022, 5, 1)
for alpha_name, alpha in dict_alphas.items():
    df_weight = alpha(dict_df_klines)
    df_weight_past = df_weight.copy()
    df_weight_recent = df_weight.copy()
    df_weight_past.loc[df_date > pd.to_datetime(past_recent_split_date), :] = 0
    df_weight_recent.loc[df_date <= pd.to_datetime(past_recent_split_date), :] = 0
    final_return, possible_maximum_drawdown = [] , []
    for df_weight, is_past in zip([df_weight_past, df_weight_recent], [True, False]):
        backtest_result = backtest.backtest_coin_strategy(df_weight, dict_df_klines, df_date, symbols)
        final_return.append(round(backtest_result['cumulative_return'].iloc[-1], 2))
        possible_maximum_drawdown.append(round(backtest_result['possible_maximum_drawdown'].min(), 2))
    print(alpha_name, 'final return', final_return, 'possible_maximum_drawdown', possible_maximum_drawdown, [f'{start_date} ~ {past_recent_split_date}', f'{past_recent_split_date} ~ {end_date}'])


print('-------------------')
print(f'future {start_date} ~ {end_date}')
for symbol in symbols:
    dict_df_klines[symbol] = backtest.get_binance_klines_data_1d(symbol, start_date, end_date, is_future=True)
for alpha_name, alpha in dict_alphas.items():
    df_weight = alpha(dict_df_klines)
    df_weight_recent = df_weight.copy()
    df_weight_recent.loc[df_date <= pd.to_datetime(past_recent_split_date), :] = 0
    backtest_result = backtest.backtest_coin_strategy(df_weight_recent, dict_df_klines, df_date, symbols)
    final_return = round(backtest_result['cumulative_return'].iloc[-1], 2)
    possible_maximum_drawdown = round(backtest_result['possible_maximum_drawdown'].min(), 2)
    print(alpha_name, 'final return', final_return, 'possible_maximum_drawdown', possible_maximum_drawdown, f'{past_recent_split_date} ~ {end_date}')

