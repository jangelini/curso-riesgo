import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.core.metrics import make_scorer

from optimization_functions import ks

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score,
                             precision_recall_curve,
                             roc_curve,
                             auc,
                             confusion_matrix,
                             PrecisionRecallDisplay,
                             RocCurveDisplay,
                             ConfusionMatrixDisplay)

ks_score = make_scorer(name='ks_score',
                       score_func=ks,
                       needs_proba=True)

if __name__=='__main__':
    df = TabularDataset('../data/full_train_pandas.parquet').drop('SK_ID_CURR', axis=1)

    train, test = train_test_split(df, random_state=42)

    hyperparameters = {
        'GBM': [
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
            {},
            'GBMLarge'
        ],
        'CAT': {},
        'XGB': {},
        'RF': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary']}}

        ],
        'XT': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary']}}
        ]
    }

    model = TabularPredictor(label='TARGET', eval_metric=ks_score)\
                        .fit(train_data=train, time_limit=18000, presets='best_quality', hyperparameters=hyperparameters)