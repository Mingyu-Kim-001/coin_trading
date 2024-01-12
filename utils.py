import duckdb
import numpy as np
import pandas as pd
from decimal import Decimal
import math
import requests
import os
import datetime


def query_on_pandas_df(str_query: str, **kwargs) -> pd.DataFrame:
  '''
  Queries a pandas DataFrame using a SQL-like syntax.
  :param str_query: A SQL-like query string to filter the DataFrame.
  :param **kwargs: Additional keyword arguments to be passed to the pandas.DataFrame.query() method.
  :return: A new DataFrame that contains the results of the query.
  '''
  for key, item in kwargs.items():
    # This is a hack to allow the query to reference pandas dataframe in the parameter.
    locals()[key] = item
  return duckdb.sql(str_query).df()


def neutralize_weight(df_weight: pd.DataFrame) -> pd.DataFrame:
  df_weight_mean = df_weight.mean(1)
  df_weight_centered = df_weight.sub(df_weight_mean, axis=0)
  df_weight_normalizer = df_weight_centered.abs().sum(1)
  df_weight_neutralized = df_weight_centered.div(df_weight_normalizer, axis=0).fillna(0)
  return df_weight_neutralized

def neutralize_weight_momentum(df_weight: pd.DataFrame) -> pd.DataFrame:
  df_weight_mean = df_weight.mean(1) * 0.9
  df_weight_centered = df_weight.sub(df_weight_mean, axis=0)
  df_weight_normalizer = df_weight_centered.abs().sum(1)
  df_weight_neutralized = df_weight_centered.div(df_weight_normalizer, axis=0).fillna(0)
  return df_weight_neutralized


def get_possible_maximum_drawdown(df_cumulative_return):
  df_cumulative_max = df_cumulative_return.cummax()
  df_drawdown = df_cumulative_return / df_cumulative_max - 1
  return df_drawdown

def get_maximum_drawdown_one_shot(df_cumulative_return):
  return df_cumulative_return.pct_change()


def round_toward_zero(x, tick_size):
  return math.floor(x / tick_size) * tick_size if x > 0 else math.ceil(x / tick_size) * tick_size

def convert_to_Decimal(number):
  return Decimal(str(number))

def trim_quantity(symbol, usdt_amount, price):
  with open('./futures_trading_rules/futures_trading_rules.csv', 'r') as f:
    df_future_trading_rules = pd.read_csv(f).set_index('symbol')
  min_qty = df_future_trading_rules.loc[symbol, 'min_qty']
  min_notional = df_future_trading_rules.loc[symbol, 'min_notional']
  quantity_trimmed = convert_to_Decimal(int(usdt_amount / price / min_qty)) * convert_to_Decimal(min_qty)
  quantity_trimmed = convert_to_Decimal(0) if abs(quantity_trimmed * convert_to_Decimal(price)) < min_notional else quantity_trimmed
  return quantity_trimmed

def trim_quantity_df(df_quantity_and_price, usdt_column_name, price_column_name):
  with open('./futures_trading_rules/futures_trading_rules.csv', 'r') as f:
    df_future_trading_rules = pd.read_csv(f).set_index('symbol')
  df_quantity_and_price = df_quantity_and_price.join(df_future_trading_rules).applymap(lambda x:Decimal(str(x)))
  df_quantity_and_price['quantity_trimmed'] = (df_quantity_and_price[usdt_column_name] / df_quantity_and_price[price_column_name] / df_quantity_and_price.min_qty).astype(int).apply(lambda x:Decimal(str(x))) * df_quantity_and_price.min_qty
  df_quantity_and_price.quantity_trimmed = np.where(
    (np.abs(df_quantity_and_price.quantity_trimmed * df_quantity_and_price[price_column_name]) < df_quantity_and_price.min_notional), Decimal(0),
    df_quantity_and_price.quantity_trimmed)
  df_quantity_and_price[f"{usdt_column_name}_trimmed"] = df_quantity_and_price.quantity_trimmed * df_quantity_and_price.price
  return df_quantity_and_price

def send_slack_message(text, slack_token, channel_name):
  requests.post("https://slack.com/api/chat.postMessage",
          headers={"Authorization": "Bearer " + slack_token},
          data={"channel": channel_name, "text": text})
  
def calculate_rsi(df_klines, lookback_n=14):
  returns = df_klines['close'].astype('float').pct_change(1)
  up = np.where(returns > 0, returns, 0)
  down = np.where(returns < 0, returns, 0)
  up_avg = up.rolling(lookback_n).mean()
  down_avg = down.abs().rolling(lookback_n).mean()
  rs = up_avg / down_avg
  rsi = 100 - (100 / (1 + rs))
  return rsi


def last_minute_of_month(year, month):
  """
  This function takes a year and a month as input and returns the last minute of that month.
  """
  # Handling the month wrapping
  if month == 12:
    next_month = 1
    next_year = year + 1
  else:
    next_month = month + 1
    next_year = year

  # The first moment of the next month
  first_moment_next_month = datetime.datetime(next_year, next_month, 1)

  # Subtracting one minute from the first moment of the next month
  last_minute_this_month = first_moment_next_month - datetime.timedelta(minutes=1)

  return last_minute_this_month


def generate_datetime_list_1min(datetime_start, datetime_end):
  """
  Generates a list of datetime objects starting from datetime_start to datetime_end
  with a frequency of 1 minute.
  """
  datetime_list = []
  current_datetime = datetime_start

  while current_datetime <= datetime_end:
    datetime_list.append(current_datetime)
    current_datetime += datetime.timedelta(minutes=1)

  return datetime_list


def get_binance_klines_data_1m(symbol, start_datetime=datetime.date, end_datetime=datetime.date, is_future=False):
  """
  Since the data is stored in monthly csv files, we need to merge them.
  """
  filename = ('f' if is_future else '') + symbol + '.csv'
  df_whole = pd.DataFrame()
  base_dir = os.path.abspath(__file__).replace('utils.py', '')
  for year_month in pd.date_range(datetime.datetime(start_datetime.year, start_datetime.month, 1, 0, 0, 0), datetime.datetime(end_datetime.year, end_datetime.month, 2, 0, 0, 0), freq='MS'):
    year, month = year_month.year, year_month.month
    month_start_datetime = datetime.datetime(year, month, 1, 0, 0, 0)
    month_end_datetime = last_minute_of_month(year, month)
    if not os.path.exists(f'{base_dir}/coin_backdata_minutely/{year}/{str(month).zfill(2)}/{filename}'):
      continue
    with open(f'{base_dir}/coin_backdata_minutely/{year}/{str(month).zfill(2)}/{filename}', 'r') as f:
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