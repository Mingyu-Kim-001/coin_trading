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
