import datetime
import pandas as pd
from utils import *
import alpha_collection_v2 as alphas
from const import FEE_RATE, SYMBOLS
import time

def get_binance_klines_data_1h(symbol, start_datetime=datetime.datetime(2017, 1, 1, 9, 0, 0),
                 end_datetime=datetime.datetime(2022, 6, 1, 0, 0, 0), freq='1h', is_future=False):
  with open(f'./coin_backdata_hourly/{"f" + symbol if is_future else symbol}.csv', 'r') as f:
    df = pd.read_csv(f)
    df['timestamp'] = df['timestamp'].apply(
      lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
  df_date_dummy = pd.DataFrame(
    {'timestamp': pd.date_range(start_datetime, end_datetime, freq=freq)})
  df_extended = df_date_dummy.merge(df, on='timestamp', how='left').fillna(0)
  for column in ['open', 'high', 'low', 'close', 'volume']:
    df_extended[column] = df_extended[column].astype(float)
  return df_extended


def get_binance_klines_data_1m(symbol, start_datetime=datetime.date, end_datetime=datetime.date, is_future=False):
  """
  Since the data is stored in monthly csv files, we need to merge them.
  """
  filename = ('f' if is_future else '') + symbol + '.csv'
  df_whole = pd.DataFrame()
  for year_month in pd.date_range(datetime.datetime(start_datetime.year, start_datetime.month, 1, 0, 0, 0), datetime.datetime(end_datetime.year, end_datetime.month, 2, 0, 0, 0), freq='MS'):
    year, month = year_month.year, year_month.month
    month_start_datetime = datetime.datetime(year, month, 1, 0, 0, 0)
    month_end_datetime = last_minute_of_month(year, month)
    with open(f'./coin_backdata_minutely/{year}/{str(month).zfill(2)}/{filename}', 'r') as f:
      df = pd.read_csv(f)
      df['timestamp'] = df['timestamp'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    
    df_datetime_dummy = pd.DataFrame({'timestamp': generate_datetime_list_1min(month_start_datetime, month_end_datetime)})
    df_extended = df_datetime_dummy.merge(df, on='timestamp', how='left').fillna(0)
    df_extended = df_extended.loc[lambda x:x['timestamp'].dt.date.between(start_datetime, end_datetime)]
    for column in ['open', 'high', 'low', 'close', 'volume']:
      df_extended[column] = df_extended[column].astype(float)
    df_whole = pd.concat([df_whole, df_extended])
  return df_whole.reset_index(drop=True)

def data_freq_convert(df:pd.DataFrame, freq:str):
  """
  Convert data frequency(in the direction of decreasing frequency)
  """
  df = df.set_index('timestamp')
  df = df.resample(freq).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
  df = df.reset_index()
  return df


def get_backtest_result(df_weight, dict_df_klines, symbols, stop_loss=-1,
            leverage=1):
  dict_df_return, dict_df_trade_size = {}, {}
  df_timestamp = list(dict_df_klines.values())[0]
  df_result = pd.DataFrame(df_timestamp, columns=[
              'timestamp'], index=df_weight.index)
  for symbol in symbols:
    df_weight_symbol = df_weight[symbol]
    df_klines = dict_df_klines[symbol].loc[lambda x:x.index.isin(
      df_weight_symbol.index)]
    pct_change = df_klines['close'].pct_change().fillna(0).replace([-np.inf, np.inf], 0)
    daily_maximum_drawdown = (dict_df_klines[symbol]['low'] - dict_df_klines[symbol]['open']) / \
      dict_df_klines[symbol]['open']

    # return calculation
    # is_stop_loss = ((df_weight_symbol < 0) & (
      # daily_maximum_drawdown < stop_loss))
    # stop_loss_return = -abs(stop_loss * df_weight_symbol)
    naive_return = pct_change * df_weight_symbol
    # dict_df_return[symbol] = pd.Series(
      # np.where(is_stop_loss, stop_loss_return, naive_return), index=df_weight.index)
    dict_df_return[symbol] = naive_return

    # trade size calculation
    # df_neutralized_weight_symbol_lag = df_weight_symbol.shift(1)
    # stop_loss_trade_size = np.where(is_stop_loss.shift(1), 2 * abs(df_weight_symbol), abs(
      # df_weight_symbol) + abs(df_neutralized_weight_symbol_lag - df_weight_symbol))
    # naive_trade_size = np.where(is_stop_loss.shift(1),
                  # abs(df_weight_symbol),
                  # abs(df_weight_symbol - df_neutralized_weight_symbol_lag))
    # dict_df_trade_size[symbol] = pd.Series(
      # np.where(is_stop_loss, stop_loss_trade_size, naive_trade_size), index=df_weight.index)
    dict_df_trade_size[symbol] = abs(df_weight_symbol.shift(1) - df_weight_symbol)

  df_result['return'] = (pd.DataFrame(dict_df_return, index=df_weight.index).sum(
    axis=1) * leverage).clip(-1, float("inf"))
  df_result['trade_size'] = pd.DataFrame(
    dict_df_trade_size, index=df_weight.index).sum(axis=1) * leverage
  df_result['fee'] = df_result['trade_size'].mul(FEE_RATE)
  df_result['return_net'] = (1 - df_result['fee']) * (1 + df_result['return']) - 1
  df_result['cumulative_fee'] = df_result['fee'].cumsum()
  df_result['cumulative_return'] = (1 + df_result['return_net']).cumprod()
  df_result['possible_maximum_drawdown'] = get_possible_maximum_drawdown(
    df_result['cumulative_return'])
  df_result['one_shot_maximum_drawdown'] = get_maximum_drawdown_one_shot(
    df_result['cumulative_return'])

  return df_result


if __name__ == '__main__':
  # get historical data
  dict_df_klines = {}
  symbols = SYMBOLS
  freq = '10T' # 10 minutes
  for symbol in symbols:
    dict_df_klines[symbol] = get_binance_klines_data_1m(symbol, datetime.date(2021, 1, 1), datetime.date(2021, 12, 25), is_future=True)
    dict_df_klines[symbol] = data_freq_convert(dict_df_klines[symbol], freq)
  # calculate weight
  alpha_names = ['control_chart_rule1', 'control_chart_rule2', 'control_chart_rule3', 'control_chart_rule4', 'control_chart_rule5', 'control_chart_rule6']
  # alpha_names = ['control_chart_rule2']
  get_weight_from_alphas = [getattr(alphas, alpha_name) for alpha_name in alpha_names]
  for get_weight_from_alpha in get_weight_from_alphas:
    df_weight = dict_df_klines[symbols[0]][['timestamp']].copy()
    for symbol in symbols:
      df_weight_symbol = get_weight_from_alpha(dict_df_klines[symbol]).rename(columns={'weight': symbol})
      df_weight_symbol[symbol] = df_weight_symbol[symbol] / len(symbols)
      df_weight = df_weight.merge(df_weight_symbol, on='timestamp', how='left')
    df_result = get_backtest_result(df_weight, dict_df_klines, symbols)
    print("final result :", df_result.cumulative_return.iloc[-1])
  