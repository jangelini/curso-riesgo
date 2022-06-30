from pyspark.sql import SparkSession, functions as F
from pyspark import SparkConf, SparkContext

from typing import List, TypeVar
from pyspark.sql.dataframe import DataFrame as SparkDf

T = TypeVar('T')

def select_and_rename(df : SparkDf, index : List[str], columns : List[str], suffix : str) -> SparkDf:
    df_k = df[index + columns]
    df_k = df_k.select(index + [F.col(column).alias(column + suffix) for column in df_k.columns if column not in index])
    return df_k

def flatten(xss : List[List[T]]) -> List[T]:
    return [x for xs in xss for x in xs]

def rename_aggregated_columns(df : SparkDf) -> SparkDf:
    return df.select([F.col(column).alias('_'.join(column.replace(')', '').split('(')[::-1])) for column in df.columns])

def aggregate(df : SparkDf, index : List[str]) -> SparkDf:
    operations = flatten([[F.max(column), F.min(column), F.mean(column), F.sum(column)] for column in df.columns if column not in index])
    aggregated = df.groupby(index).agg(*operations)
    aggregated = rename_aggregated_columns(aggregated)
    return aggregated

def get_dummies(df : SparkDf, indexes : List[str], column : str) -> SparkDf:    
    values      = df.select(column).distinct().rdd.flatMap(lambda x: x).collect()
    values_expr = [F.when(F.col(column)==value, 1).otherwise(0).alias(value) for value in values]
    return df.select(*indexes, *values_expr)