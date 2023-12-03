from datetime import datetime
from datetime import timedelta
import pandas as pd
import requests
from typing import *
import time

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
            return None

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

def ms_to_dt_utc(ms: int) -> datetime:
    return datetime.utcfromtimestamp(ms / 1000)

def ms_to_dt_local(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000)

def GetDataFrame(data):
    df = pd.DataFrame(data, columns=['timestamp', "open", "high", "low", "close", "volume"])
    df["timestamp"] = df["timestamp"].apply(lambda x: ms_to_dt_local(x))
    df['date'] = df["timestamp"].dt.strftime("%d/%m/%Y")
    df['time'] = df["timestamp"].dt.strftime("%H:%M:%S")
    column_names = ["date", "time", "open", "high", "low", "close", "volume"]
    df = df.set_index('timestamp')
    df = df.reindex(columns=column_names)

    return df

def GetHistoricalData(client, symbol, start_time, end_time, interval='1m', limit=1500):
    collection = []

    while start_time < end_time:
        data = client.get_historical_data(symbol, start_time=start_time, end_time=end_time, interval=interval, limit=limit)
        if len(data) == 0:
            return collection
        start_time = int(data[-1][0] + 1000)
        collection +=data
        time.sleep(0.001)
        print(symbol, datetime.fromtimestamp(start_time/1000))
    return collection

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


interval = "1m"
for is_future in [True]:
    client = BinanceClient(futures=is_future)
    for symbol in ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'MATICUSDT', 'SOLUSDT', 'LTCUSDT', 'TRXUSDT', 'DOTUSDT', 'BNBUSDT', 'BCHUSDT', 'XLMUSDT', '1000SHIBUSDT']:
        start_date = datetime(2023, 6, 1, 0, 0)
        end_date = datetime(2023, 11, 30, 9, 0)
        fromDate = int(start_date.timestamp() * 1000)
        toDate = int(end_date.timestamp() * 1000)
        data = GetHistoricalData(client, symbol, fromDate, toDate, interval=interval)
        df = GetDataFrame(data)
        try:
            existing_data = pd.read_csv(f"./coin_backdata_hourly/{'f' if is_future else ''}{symbol}.csv").set_index('timestamp')
        except:
            existing_data = pd.DataFrame(columns=['date', 'time', 'open', 'high', 'low', 'close', 'volume', 'timestamp']).set_index('timestamp')
        new_rows = df[df.index > (existing_data.index.max() if len(existing_data) > 0 else pd.to_datetime(0))]
        updated_data = pd.concat([existing_data, new_rows])
        updated_data.to_csv(f"./coin_backdata_hourly/{'f' if is_future else ''}{symbol}.csv")