import os

import pandas as pd
import numpy as np
from functools import partial

from pyspark.sql import SparkSession, functions as F

from pyspark import SparkConf, SparkContext

from typing import List, TypeVar
from pyspark.sql.dataframe import DataFrame as SparkDf


# master("local[*]") hace que se usen todos los nucleos de la compu
# config("spark.driver.memory", "12g") esto indica que le suba la memoria que usa
spark = SparkSession.builder.master("local[*]").config("spark.driver.memory", "12g").appName('credito').getOrCreate()
spark


DATA_PATH = os.path.join('data')
INDEX     = ['SK_ID_CURR']


bureau = spark.read.csv(os.path.join(DATA_PATH, 'bureau.csv'), header=True)


bureau_consumer = bureau.filter(bureau['CREDIT_TYPE']=='Consumer credit')
bureau_card = bureau.filter(bureau['CREDIT_TYPE']=='Credit card')
bureau_car = bureau.filter(bureau['CREDIT_TYPE']=='Car loan')
bureau_mort = bureau.filter(bureau['CREDIT_TYPE']=='Mortgage')


bureau_consumer_k = bureau_consumer.select("SK_ID_CURR",
                                            "CREDIT_DAY_OVERDUE",
                                             "AMT_CREDIT_MAX_OVERDUE",
                                             "AMT_CREDIT_SUM",
                                             "AMT_CREDIT_SUM_DEBT")

bureau_card_k = bureau_card.select("SK_ID_CURR",
                                            "CREDIT_DAY_OVERDUE",
                                             "AMT_CREDIT_MAX_OVERDUE",
                                             "AMT_CREDIT_SUM",
                                             "AMT_CREDIT_SUM_DEBT")

bureau_car_k = bureau_car.select("SK_ID_CURR",
                                            "CREDIT_DAY_OVERDUE",
                                             "AMT_CREDIT_MAX_OVERDUE",
                                             "AMT_CREDIT_SUM",
                                             "AMT_CREDIT_SUM_DEBT")

bureau_mort_k = bureau_mort.select("SK_ID_CURR",
                                            "CREDIT_DAY_OVERDUE",
                                             "AMT_CREDIT_MAX_OVERDUE",
                                             "AMT_CREDIT_SUM",
                                             "AMT_CREDIT_SUM_DEBT")


 def flatten(xss):
    return [x for xs in xss for x in xs]

def aggregate(df, index):
    aggregations = flatten([[F.sum(column), 
                             F.max(column), 
                             F.min(column), 
                             F.mean(column)] for column in df.columns if column != index])
    return df.groupby(index).agg(*aggregations)



bureau_consumer_k_agg = bureau_consumer_k.transform(partial(aggregate, index="SK_ID_CURR"))
bureau_card_k_agg = bureau_card_k.transform(partial(aggregate, index="SK_ID_CURR"))
bureau_car_k_agg = bureau_car_k.transform(partial(aggregate, index="SK_ID_CURR"))
bureau_mort_k_agg = bureau_mort_k.transform(partial(aggregate, index="SK_ID_CURR"))