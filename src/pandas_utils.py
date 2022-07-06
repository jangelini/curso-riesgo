from pandas import DataFrame

from typing import List

def select_and_rename(df : DataFrame, index : List[str], columns : List[str], suffix : str) -> DataFrame:
    """
    This function selects a list of dataframe columns and renames them according to a given suffix, excluding the index column.

    Args:
        df (DataFrame): Input dataframe.
        index (List[str]): A list containing the name of the index column.
        columns (List[str]): A list containing the names of the remaining columns.
        suffix (str): The suffix to add to the columns.

    Returns:
        DataFrame: A dataframe that contains the untouched index column and the renamed columns.
    """
    df_k         = df[index + columns]
    df_k.columns = index + [f'{column}_{suffix}' for column in df_k.columns if column!=index[0]]
    return df_k

def aggregate(df : DataFrame, index : List[str]) -> DataFrame:
    """
    This function takes a dataframe and aggregates it according to a provided index.
    The aggregations performed are 'count', 'max', 'min', 'median', 'mean' and 'sum'.

    Args:
        df (DataFrame): Input dataframe.
        index (List[str]): A list containing the name of the index column.

    Returns:
        DataFrame: The aggregated dataframe.
    """
    aggregated         = df.groupby(index).agg(['count', 'max', 'min', 'median', 'mean', 'sum'])
    aggregated.columns = ['_'.join(column) for column in aggregated.columns]
    return aggregated