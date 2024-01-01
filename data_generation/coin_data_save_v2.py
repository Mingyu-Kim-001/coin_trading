import datetime
import pandas as pd
import requests
from typing import *
import time
from utils import *
from const import SYMBOLS

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class BinanceClient:
  def __init__(self, futures=False):
    self.exchange = "BINANCE"
    self.futures = futures

    if self.futures:
      self._base_url = "https://fapi.binance.com"
    else:
      self._base_url = "https://api.binance.com"

    self.symbols = self._get_symbols()

  def _make_request(self, endpoint: str, query_parameters: Dict):
    try:
      response = requests.get(self._base_url + endpoint, params=query_parameters)
    except Exception as e:
      print("Connection error while making request to %s: %s", endpoint, e)
      raise e

    if response.status_code == 200:
      return response.json()
    else:
      print("Error while making request to %s: %s (status code = %s)",
             endpoint, response.json(), response.status_code)
      return None

  def _get_symbols(self) -> List[str]:

    params = dict()

    endpoint = "/fapi/v1/exchangeInfo" if self.futures else "/api/v3/exchangeInfo"
    data = self._make_request(endpoint, params)

    symbols = [x["symbol"] for x in data["symbols"]]

    return symbols

  def get_historical_data(self, symbol: str, interval: Optional[str] = "1m", start_time: Optional[int] = None, end_time: Optional[int] = None, limit: Optional[int] = 1500):

    params = dict()

    params["symbol"] = symbol
    params["interval"] = interval
    params["limit"] = limit

    if start_time is not None:
      params["startTime"] = start_time
    if end_time is not None:
      params["endTime"] = end_time

    endpoint = "/fapi/v1/klines" if self.futures else "/api/v3/klines"
    raw_candles = self._make_request(endpoint, params)

    candles = []

    if raw_candles is not None:
      for c in raw_candles:
        candles.append((float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5]),))
      return candles
    else:
      return None

def ms_to_dt_utc(ms: int) -> datetime.datetime:
  return datetime.datetime.utcfromtimestamp(ms / 1000)

def ms_to_dt_local(ms: int) -> datetime.datetime:
  return datetime.datetime.fromtimestamp(ms / 1000)

def GetDataFrame(data):
  df = pd.DataFrame(data, columns=['timestamp', "open", "high", "low", "close", "volume"])
  if len(df) == 0:
    return df
  df["timestamp"] = df["timestamp"].apply(lambda x: ms_to_dt_local(x))
  column_names = ["open", "high", "low", "close", "volume"]
  df = df.set_index('timestamp')
  df = df.reindex(columns=column_names)
  return df

def GetHistoricalData(client, symbol, start_time, end_time, interval='1m', limit=1500):
  collection = []
  while start_time < end_time:
    max_try_cnt = 5
    try_cnt = 0
    while try_cnt < max_try_cnt:
      try:
        data = client.get_historical_data(symbol, start_time=start_time, end_time=end_time, interval=interval, limit=limit)
        break
      except:
        try_cnt += 1
        time.sleep(10)

    if not data or len(data) == 0:
      return collection
    start_time = int(data[-1][0] + 1000)
    collection +=data
    time.sleep(0.001)
    print(symbol, datetime.datetime.fromtimestamp(start_time/1000))
  return collection


def run(symbol:str, is_future:bool, year:int, month:int, freq='1m', is_overwrite=False):
  client = BinanceClient(futures=is_future)
  month_start_datetime = datetime.datetime(year, month, 1, 0, 0, 0)
  month_end_datetime = last_minute_of_month(year, month)
  start_timestamp = int(month_start_datetime.timestamp() * 1000)
  end_timestamp = int(month_end_datetime.timestamp() * 1000)
  if is_overwrite:
    print(f"Overwriting {symbol} {year}/{str(month).zfill(2)}")
    data = GetHistoricalData(client, symbol, start_timestamp, end_timestamp, interval=freq)
    updated_data = GetDataFrame(data)
  else:
    try:
      existing_data = pd.read_csv(f"./coin_backdata_minutely/{year}/{str(month).zfill(2)}/{'f' if is_future else ''}{symbol}.csv").set_index('timestamp')
    except:
      existing_data = pd.DataFrame(columns=['date', 'time', 'open', 'high', 'low', 'close', 'volume', 'timestamp']).set_index('timestamp')
    existing_latest_timestamp = datetime.datetime.strptime(existing_data.index.max(), '%Y-%m-%d %H:%M:%S') if len(existing_data) > 0 else None
    if existing_latest_timestamp < month_end_datetime or existing_latest_timestamp is None:
      data = GetHistoricalData(client, symbol, start_timestamp, end_timestamp, interval=freq)
      df = GetDataFrame(data)
      new_rows = df[df.index > existing_latest_timestamp]
      updated_data = pd.concat([existing_data, new_rows])
    else:
      print(f"Skipping {symbol} {year}/{str(month).zfill(2)} since it is already up-to-date.")
      return
  if len(updated_data) == 0:
    return
  if not os.path.exists(f"./coin_backdata_minutely/{year}/{str(month).zfill(2)}"):
    os.makedirs(f"./coin_backdata_minutely/{year}/{str(month).zfill(2)}")
  print("write to", f"./coin_backdata_minutely/{year}/{str(month).zfill(2)}/{'f' if is_future else ''}{symbol}.csv")
  updated_data.to_csv(f"./coin_backdata_minutely/{year}/{str(month).zfill(2)}/{'f' if is_future else ''}{symbol}.csv")
        
      

if __name__ == '__main__':
  interval = "1m"
  is_overwrite = False
  start_year_month = '2023-12'
  end_year_month = '2023-12'
  for is_future in [False, True]:
    for symbol in SYMBOLS:
      for year_month in pd.date_range(datetime.datetime.strptime(start_year_month, '%Y-%m'), datetime.datetime.strptime(end_year_month, '%Y-%m'), freq='MS'):
        year, month = int(year_month.year), int(year_month.month)
        run(symbol, is_future, year, month, freq='1m', is_overwrite=is_overwrite)

  