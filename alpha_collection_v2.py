import pandas as pd
from utils import *
from run_backtest_v2 import get_binance_klines_data_1m
from pathlib import Path


def close_position_in_nday_bollinger_band_ewm(df_kline: dict, window=110, weight_max=None, shift=1):
  """
  Posiiton of close price in n-timewindow Bollinger Band Exponential Weighted Moving Average (EWMA).
  """
  df_weight = df_kline['close'].astype('float').sub(df_kline['close'].astype('float').ewm(span=window, adjust=False).mean()).div(df_kline['close'].astype('float').ewm(span=window, adjust=False).std()).shift(shift)
  return df_weight


# control chart - https://wire.insiderfinance.io/trading-the-stock-market-in-an-unconventional-way-using-control-charts-f6e9aca3d8a0
def control_chart_rule1(df_kline, window=10, shift=1):
  """
  One point beyond the 3 stdev control limit
  """
  df_weight = query_on_pandas_df(f"""
    --sql
    WITH sma_added
            AS (SELECT *,
                        AVG(close) OVER (ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW)        AS sma,
                        3 * STDDEV(close) OVER (ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW) AS std3
                FROM df)
    SELECT timestamp, LAG(CASE WHEN close < sma - std3 THEN 1 WHEN close > sma + std3 THEN -1 ELSE 0 END, {shift}) OVER (ORDER BY timestamp) AS weight,
    FROM sma_added;
  """,
  df=df_kline)
  return df_weight

def control_chart_rule2(df_kline, window=10, shift=1):
  """
  Eight or more points on one side of the centerline without crossing
  """
  df_weight = query_on_pandas_df(f"""
    --sql
    WITH sma_added AS
            (SELECT *,
                    AVG(close) OVER (ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW) AS sma
              FROM df),
        count_upper_and_lower AS
            (SELECT *,
                    SUM(IF(close > sma, 1, 0))
                        OVER (ORDER BY timestamp ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS count_upper,
                    SUM(IF(close > sma, 1, 0))
                        OVER (ORDER BY timestamp ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS count_lower
              FROM sma_added)
    SELECT timestamp,
          LAG(CASE WHEN count_upper >= 8 THEN 1 WHEN count_lower >= 8 THEN -1 ELSE 0 END, {shift}) OVER () AS weight
    FROM count_upper_and_lower;
  """,
  df=df_kline)
  return df_weight

def control_chart_rule3(df_kline, window=10, shift=1):
    """
    Four out of five points over 1 stdev -> -1, under -1 stdev -> 1
    """
    df_weight = query_on_pandas_df(f"""
      --sql
      WITH sma_added AS
              (SELECT *,
                      AVG(close) OVER (ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW)    AS sma,
                      STDDEV(close) OVER (ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW) AS std
                FROM df),
          count_cnt_out_of_five AS
              (SELECT *,
                      SUM(IF(close < sma - std, 1, 0))
                          OVER (ORDER BY timestamp ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS count_lower,
                      SUM(IF(close > sma + std, 1, 0))
                          OVER (ORDER BY timestamp ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS count_upper
                FROM sma_added)
      SELECT timestamp,
            LAG(CASE WHEN count_lower >= 4 THEN 1 WHEN count_upper >= 4 THEN -1 ELSE 0 END, {shift}) OVER () AS weight
      FROM count_cnt_out_of_five;
    """,
    df=df_kline)
    return df_weight

def control_chart_rule4(df_kline, window=10, shift=1):
  """
  Six points or more in a row steadily increasing -> -1, decreasing -> 1
  """
  df_weight = query_on_pandas_df(f"""
    --sql
    WITH steadily_changing AS
            (SELECT timestamp,
                    CASE
                        WHEN close < LAG(close, 1) OVER (ORDER BY timestamp) AND
                              LAG(close, 1) OVER (ORDER BY timestamp) < LAG(close, 2) OVER (ORDER BY timestamp) AND
                              LAG(close, 2) OVER (ORDER BY timestamp) < LAG(close, 3) OVER (ORDER BY timestamp) AND
                              LAG(close, 3) OVER (ORDER BY timestamp) < LAG(close, 4) OVER (ORDER BY timestamp) AND
                              LAG(close, 4) OVER (ORDER BY timestamp) < LAG(close, 5) OVER (ORDER BY timestamp) 
                             THEN 1
                        WHEN close > LAG(close, 1) OVER (ORDER BY timestamp) AND
                              LAG(close, 1) OVER (ORDER BY timestamp) > LAG(close, 2) OVER (ORDER BY timestamp) AND
                              LAG(close, 2) OVER (ORDER BY timestamp) > LAG(close, 3) OVER (ORDER BY timestamp) AND
                              LAG(close, 3) OVER (ORDER BY timestamp) > LAG(close, 4) OVER (ORDER BY timestamp) AND
                              LAG(close, 4) OVER (ORDER BY timestamp) > LAG(close, 5) OVER (ORDER BY timestamp) 
                             THEN -1
                        ELSE 0 END AS steady_change
            FROM df)
    SELECT timestamp,
          LAG(steady_change, {shift}) OVER (ORDER BY timestamp) AS weight
    FROM steadily_changing;
  """,
  df=df_kline)
  return df_weight


def control_chart_rule5(df_kline, window=10, shift=1):
  """
  Two out of three points over 2 stdev -> -1, under -2 stdev -> 1
  """
  df_weight = query_on_pandas_df(f"""
    --sql
    WITH sma_added AS
            (SELECT *,
                    AVG(close) OVER (ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW)    AS sma,
                    STDDEV(close) OVER (ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW) AS std
              FROM df),
        count_cnt_out_of_three AS
            (SELECT *,
                    SUM(IF(close < sma - 2 * std, 1, 0))
                        OVER (ORDER BY timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS count_lower,
                    SUM(IF(close > sma + 2 * std, 1, 0))
                        OVER (ORDER BY timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS count_upper
              FROM sma_added)
    SELECT timestamp,
          LAG(CASE WHEN count_lower >= 2 THEN 1 WHEN count_upper >= 2 THEN -1 ELSE 0 END, {shift}) OVER () AS weight
    FROM count_cnt_out_of_three;
  """,
  df=df_kline)
  return df_weight

def control_chart_rule6(df_kline, window=10, shift=1):
  """
  14 points in a row with different zone.
  zone is defined by 1, 2 stdev and sma
  """
  df_weight = query_on_pandas_df(f"""
    --sql
    WITH sma_added AS
            (SELECT *,
                    AVG(close) OVER (ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW)    AS sma,
                    STDDEV(close) OVER (ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW) AS std
              FROM df),
        zone AS
            (SELECT *,
                    LAG(CASE WHEN close > sma + 2 * std THEN 3
                         WHEN close > sma + std THEN 2
                         WHEN close > sma THEN 1
                         WHEN close > sma - std THEN -1
                         WHEN close > sma - 2 * std THEN -2
                         ELSE -3 END) OVER (ORDER BY timestamp) AS prev_zone,
                    CASE WHEN close > sma + 2 * std THEN 3
                         WHEN close > sma + std THEN 2
                         WHEN close > sma THEN 1
                         WHEN close > sma - std THEN -1
                         WHEN close > sma - 2 * std THEN -2
                         ELSE -3 END AS zone
              FROM sma_added),
        count_cnt_out_of_fourteen AS
            (SELECT *,
                    SUM(IF(zone <> prev_zone, 0, 1))
                        OVER (ORDER BY timestamp ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS count_diff
              FROM zone)
    SELECT timestamp,
          LAG(CASE WHEN count_diff >= 14 THEN 1 ELSE -1 END, {shift}) OVER (ORDER BY timestamp) AS weight
    FROM count_cnt_out_of_fourteen;
  """,
  df=df_kline)
  return df_weight


def pairs_trading_with_svr_offilne(year, symbol):
  df_weight = pd.DataFrame()
  offline_data_path = Path(__file__).parent / 'alpha_weight_offline' / 'pairs_trading_svr'
  # df_symbol = pd.read_csv(f'./pairs_trading_with_svr_data/data_raw_{symbol}_2023.csv')
  df_symbol = pd.read_csv(offline_data_path / f'data_raw_{symbol}_{year}.csv')
  df_symbol = df_symbol.loc[lambda x:x.timestamp.str.contains(f'{year}')]
  df_symbol_weight = df_symbol[['vote', 'timestamp']]
  return df_symbol_weight

    

# for debug
if __name__ == '__main__':
  df_debug = get_binance_klines_data_1m('BTCUSDT', datetime.date(2022, 1, 1), datetime.date(2023, 1, 2), is_future=True)
  df_weight = control_chart_rule4(df_debug)