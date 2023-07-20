import duckdb
import numpy as np
import pandas as pd
from decimal import Decimal
import math
import requests
import os


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


def trim_quantity(df_quantity_and_price, usdt_column_name, price_column_name):
    with open('./futures_trading_rules/futures_trading_rules.csv', 'r') as f:
        df_future_trading_rules = pd.read_csv(f).set_index('symbol')
    df_quantity_and_price = df_quantity_and_price.join(df_future_trading_rules).applymap(lambda x:Decimal(str(x)))
    df_quantity_and_price['quantity_trimmed'] = (df_quantity_and_price[usdt_column_name] / df_quantity_and_price[price_column_name] / df_quantity_and_price.min_qty).astype(int).apply(lambda x:Decimal(str(x))) * df_quantity_and_price.min_qty
    df_quantity_and_price.quantity_trimmed = np.where(
        (np.abs(df_quantity_and_price.quantity_trimmed * df_quantity_and_price[price_column_name]) < df_quantity_and_price.min_notinoal), Decimal(0),
        df_quantity_and_price.quantity_trimmed)
    df_quantity_and_price[f"{usdt_column_name}_trimmed"] = df_quantity_and_price.quantity_trimmed * df_quantity_and_price.price
    return df_quantity_and_price

def send_slack_message(text, slack_token, channel_name):
    requests.post("https://slack.com/api/chat.postMessage",
                  headers={"Authorization": "Bearer " + slack_token},
                  data={"channel": channel_name, "text": text})