import datetime
import pandas as pd
from coin_trading_backtest import market_neutral_trading_backtest_binance
from alpha_collection import Alphas

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
backtest = market_neutral_trading_backtest_binance()
alpha_collection = Alphas()
alpha_org_names = [alpha_name for alpha_name in alpha_collection.__dir__() if not alpha_name.startswith('_')]
dict_alphas = {}
for alpha_name in alpha_org_names:
    if 'nday' in alpha_name:
        for n in list(range(1, 5)) + [10, 20, 50, 100, 200]:
            dict_alphas[alpha_name + f'_{n}'] = (lambda name, n: lambda x: getattr(alpha_collection, name)(x, n))(alpha_name, n)
    else:
        dict_alphas[alpha_name] = getattr(alpha_collection, alpha_name)
# for alpha_name in alpha_org_names:
#     dict_alphas[alpha_name] = getattr(alpha_collection, alpha_name)


dict_df_klines = {}
start_date = datetime.date(2017, 8, 17)
end_date = datetime.date(2022, 5, 1)
print(f'past {start_date} ~ {end_date}')
symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'DOGEUSDT', 'LTCUSDT', 'MATICUSDT', 'TRXUSDT', 'ADAUSDT', 'SOLUSDT']
# symbols = ['BTCUSDT', 'ETHUSDT']

for symbol in symbols:
    dict_df_klines[symbol] = backtest.get_binance_klines_data_1d(symbol, start_date, end_date)
df_date = list(dict_df_klines.values())[0]['date']
df_close = pd.concat(
    [df_klines['close'].astype('float').rename(f'{symbol}_close') for symbol, df_klines in dict_df_klines.items()],
    axis=1)
df_low = pd.concat(
    [df_klines['low'].astype('float').rename(f'{symbol}_low') for symbol, df_klines in dict_df_klines.items()],
    axis=1)

for alpha_name, alpha in dict_alphas.items():
    df_weight = alpha(dict_df_klines)
    # backtest_result = backtest.backtest_coin_strategy(df_weight, df_date, df_close, df_low, symbols, stop_loss=-0.05)
    backtest_result = backtest.backtest_coin_strategy(df_weight, dict_df_klines, df_date, symbols)
    final_return = backtest_result['cumulative_return'].iloc[-1]
    possible_maximum_drawdown = backtest_result['possible_maximum_drawdown'].min()
    print(alpha_name, 'final return', round(final_return, 2), 'possible_maximum_drawdown', round(possible_maximum_drawdown, 2))

start_date = datetime.date(2022, 5, 2)
end_date = datetime.date(2023, 5, 1)
is_future = True
print(f'recent {start_date} ~ {end_date}')
for symbol in symbols:
    dict_df_klines[symbol] = backtest.get_binance_klines_data_1d(symbol, start_date, end_date, is_future)
df_date = list(dict_df_klines.values())[0]['date']
df_close = pd.concat(
    [df_klines['close'].astype('float').rename(f'{symbol}_close') for symbol, df_klines in dict_df_klines.items()],
    axis=1)
df_low = pd.concat(
    [df_klines['low'].astype('float').rename(f'{symbol}_low') for symbol, df_klines in dict_df_klines.items()],
    axis=1)

for alpha_name, alpha in dict_alphas.items():
    df_weight = alpha(dict_df_klines)
    backtest_result = backtest.backtest_coin_strategy(df_weight, df_date, df_close, df_low, symbols, stop_loss=-0.05)
    final_return = backtest_result['cumulative_return'].iloc[-1]
    possible_maximum_drawdown = backtest_result['possible_maximum_drawdown'].min()
    print(alpha_name, 'final return', round(final_return, 2), 'possible_maximum_drawdown', round(possible_maximum_drawdown, 2))