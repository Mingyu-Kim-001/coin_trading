import pandas as pd
import duckdb

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


def neutralize_weight(df_weight: pd.DataFrame):
    df_weight_mean = df_weight.mean(1)
    df_weight_neutralized = df_weight.sub(df_weight_mean, axis=0)
    df_weight_normalizer = df_weight_neutralized.abs().sum(1)
    df_weight_normalized = df_weight_neutralized.div(df_weight_normalizer, axis=0).fillna(0)
    return df_weight_normalized

def get_possible_maximum_drawdown(df_cumulative_return):
    df_cumulative_max = df_cumulative_return.cummax()
    df_drawdown = df_cumulative_return / df_cumulative_max - 1
    return df_drawdown