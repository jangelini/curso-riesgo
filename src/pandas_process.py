import os
from functools import partial
from argparse import ArgumentParser

import pandas as pd
import numpy as np

import pandas_utils as utils

parser = ArgumentParser(description='Read, process and save the datasets.')
parser.add_argument('--data', type=str, help='The path of the directory containing the data.')
parser.add_argument('--index', help='The name of the index column.')
args = parser.parse_args()

DATA_PATH = args.data    #os.path.join('..', 'data')
INDEX     = [args.index] #['SK_ID_CURR']

if __name__=='__main__':
    
    appl           = pd.read_parquet(os.path.join(DATA_PATH, 'application_train.parquet'))
    bureau         = pd.read_parquet(os.path.join(DATA_PATH, 'bureau.parquet'))
    bureau_balance = pd.read_parquet(os.path.join(DATA_PATH, 'bureau_balance.parquet'))
    card_balance   = pd.read_parquet(os.path.join(DATA_PATH, 'credit_card_balance.parquet'))
    installments   = pd.read_parquet(os.path.join(DATA_PATH, 'installments_payments.parquet'))
    prev           = pd.read_parquet(os.path.join(DATA_PATH, 'previous_application.parquet'))
    pos_cash       = pd.read_parquet(os.path.join(DATA_PATH, 'POS_CASH_balance.parquet'))

    bureau_consumer = bureau.loc[(bureau['CREDIT_TYPE']=='Consumer credit')]
    bureau_card     = bureau.loc[(bureau['CREDIT_TYPE']=='Credit card')]
    bureau_car      = bureau.loc[(bureau['CREDIT_TYPE']=='Car loan')]
    bureau_mort     = bureau.loc[(bureau['CREDIT_TYPE']=='Mortgage')]
    
    bureau_columns = ["CREDIT_DAY_OVERDUE", "AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT"]
    
    bureau_consumer_agg = utils.select_and_rename(bureau_consumer, INDEX, bureau_columns, 'co').pipe(partial(utils.aggregate, index=INDEX))
    bureau_car_agg      = utils.select_and_rename(bureau_car, INDEX, bureau_columns, 'ca').pipe(partial(utils.aggregate, index=INDEX))
    bureau_mort_agg     = utils.select_and_rename(bureau_mort, INDEX, bureau_columns, 'mo').pipe(partial(utils.aggregate, index=INDEX))
    bureau_card_agg     = utils.select_and_rename(bureau_card, INDEX, bureau_columns, 'cd').pipe(partial(utils.aggregate, index=INDEX))
    
    appl_bureau = (
        appl.merge(bureau_consumer_agg, how='left', on=INDEX, sort="True")
            .merge(bureau_mort_agg,     how='left', on=INDEX, sort="True")
            .merge(bureau_car_agg,      how='left', on=INDEX, sort="True")
            .merge(bureau_card_agg,     how='left', on=INDEX, sort="True")
            .set_index(INDEX)
    )
    
    bureau_balance_clean = (bureau_balance.drop('MONTHS_BALANCE', axis=1)
                                          .merge(bureau[INDEX + ['SK_ID_BUREAU']],
                                                 how='left',
                                                 left_on=['SK_ID_BUREAU'],
                                                 right_on=['SK_ID_BUREAU'],
                                                 sort="True")
                                          .pipe(pd.get_dummies, columns=['STATUS'], prefix_sep="_", sparse=True)
                                          .groupby(INDEX).sum()
                                          .reset_index())
    
    appl_bureau = appl_bureau.merge(
        bureau_balance_clean,
        how="left", left_on=INDEX, right_on=INDEX, sort="True"
    )
    
    appl_bureau.to_dense().to_parquet(os.path.join(DATA_PATH, 'full_train_pandas.parquet'))