from functools import partial

from pandas import DataFrame
import pandas as pd
import numpy as np

from typing import List

def select_and_rename(df : DataFrame, index : List[str], columns : List[str], suffix : str) -> DataFrame:
    df_k         = df[index + columns]
    df_k.columns = index + [f'{column}_{suffix}' for column in df_k.columns if column!=index[0]]
    return df_k

def aggregate(df : DataFrame, index : List[str]) -> DataFrame:
    aggregated         = df.groupby(index).agg(['count', 'max', 'min', 'median', 'mean', 'sum'])
    aggregated.columns = ['_'.join(column) for column in aggregated.columns]
    return aggregated