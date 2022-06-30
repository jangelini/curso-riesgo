import os
from argparse import ArgumentParser

import pandas as pd
import numpy as np
from functools import partial

from pyspark.sql import SparkSession, functions as F
from pyspark import SparkConf, SparkContext

import spark_utils as utils

from typing import List, TypeVar
from pyspark.sql.dataframe import DataFrame as SparkDf

parser = ArgumentParser(description='Read, process and save the datasets.')
parser.add_argument('--data', type=str, help='The path of the directory containing the data.')
parser.add_argument('--index', help='The name of the index column.')
args = parser.parse_args()

DATA_PATH = args.data    #os.path.join('..', 'data')
INDEX     = [args.index] #['SK_ID_CURR']

if __name__=='__main__':

    spark = SparkSession.builder.master("local[*]").config("spark.driver.memory", "12g").appName('credito').getOrCreate()

    appl           = spark.read.csv(os.path.join(DATA_PATH, 'application_train.csv'),     header=True)
    bureau         = spark.read.csv(os.path.join(DATA_PATH, 'bureau.csv'),                header=True)
    bureau_balance = spark.read.csv(os.path.join(DATA_PATH, 'bureau_balance.csv'),        header=True)
    card_balance   = spark.read.csv(os.path.join(DATA_PATH, 'credit_card_balance.csv'),   header=True)
    installments   = spark.read.csv(os.path.join(DATA_PATH, 'installments_payments.csv'), header=True)
    prev           = spark.read.csv(os.path.join(DATA_PATH, 'previous_application.csv'),  header=True)
    pos_cash       = spark.read.csv(os.path.join(DATA_PATH, 'POS_CASH_balance.csv'),      header=True)

    bureau_consumer = bureau.filter(bureau['CREDIT_TYPE']=='Consumer credit')
    bureau_card     = bureau.filter(bureau['CREDIT_TYPE']=='Credit card')
    bureau_car      = bureau.filter(bureau['CREDIT_TYPE']=='Car loan')
    bureau_mort     = bureau.filter(bureau['CREDIT_TYPE']=='Mortgage')


    bureau_columns    = ["CREDIT_DAY_OVERDUE", "AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT"]

    bureau_consumer_k = bureau_consumer.transform(partial(utils.select_and_rename, index=INDEX, columns=bureau_columns, suffix='_co'))
    bureau_car_k      = bureau_car.transform(partial(utils.select_and_rename,      index=INDEX, columns=bureau_columns, suffix='_ca'))
    bureau_mort_k     = bureau_mort.transform(partial(utils.select_and_rename,     index=INDEX, columns=bureau_columns, suffix='_mo'))
    bureau_card_k     = bureau_card.transform(partial(utils.select_and_rename,     index=INDEX, columns=bureau_columns, suffix='_cd'))


    bureau_consumer_agg = bureau_consumer_k.transform(partial(utils.aggregate, index=INDEX))
    bureau_car_agg      = bureau_car_k.transform(partial(utils.aggregate,      index=INDEX))
    bureau_mort_agg     = bureau_mort_k.transform(partial(utils.aggregate,     index=INDEX))
    bureau_card_agg     = bureau_card_k.transform(partial(utils.aggregate,     index=INDEX))


    appl_bureau = (
        appl.join(bureau_consumer_agg, how='left', on=INDEX)
            .join(bureau_car_agg,      how='left', on=INDEX)
            .join(bureau_mort_agg,     how='left', on=INDEX)
            .join(bureau_card_agg,     how='left', on=INDEX)
    )



    appl_bureau_history = appl_bureau.join(
                            bureau_balance.drop('MONTHS_BALANCE')
                                          .join(bureau.select(INDEX + ['SK_ID_BUREAU']), how='left', on=['SK_ID_BUREAU'])
                                          .transform(partial(utils.get_dummies, indexes=INDEX + ['SK_ID_BUREAU'], column='STATUS'))
                                          .groupBy(INDEX)
                                          .sum()
                                          .transform(utils.rename_aggregated_columns),
                            how='left', on=INDEX
    )

    

    appl_bureau_history_prev = appl_bureau_history.join(utils.aggregate(prev.select(['SK_ID_CURR', 'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT']), INDEX), on=INDEX)



    appl_bureau_history_prev_card = appl_bureau_history_prev.join(card_balance.select(
                        INDEX + [column for column in card_balance.columns if 'amt' in column.lower() or 'cnt' in column.lower() or 'dpd' in column.lower()]
                    )
                                        .transform(partial(utils.aggregate, index=INDEX)),
                                             how='left', on=INDEX)


    to_drop_pos = ['NAME_CONTRACT_STATUS', 'SK_ID_PREV']

    appl_bureau_history_prev_card_pos = appl_bureau_history_prev_card.join(
        pos_cash.drop(*to_drop_pos)
                .transform(partial(utils.select_and_rename, index=INDEX, columns=[column for column in pos_cash.columns if column not in to_drop_pos], suffix='pos'))
                .transform(partial(utils.aggregate, index=INDEX)),
        how='left', on=INDEX
    )



    appl_bureau_history_prev_card_pos_inst = appl_bureau_history_prev_card_pos.join(
        installments.drop('SK_ID_PREV').transform(partial(utils.aggregate, index=INDEX)),
        on=INDEX, how='left'
    )



    appl_bureau_history_prev_card_pos_inst.write.parquet(os.path.join(DATA_PATH, 'full_train'))