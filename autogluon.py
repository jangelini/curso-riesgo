import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor, TabularDataset

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

if __name__=='__main__':
    df = TabularDataset('../data/full_train_pandas.csv')

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
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},

        ],
        'XT': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
        ]
    }

    model = TabularPredictor(label='HasDetections', eval_metric='average_precision')\
                        .fit(train_data=train, time_limit=36000, presets='best_quality', hyperparameters=hyperparameters)
    
    model.leaderboard().to_parquet('autogluon_leaderboard.parquet')

    probabilities = model.predict_proba(test.drop('TARGET', axis=1))

    fpr, tpr, thresholds = roc_curve(test['TARGET'], probabilities[1])
    roc_auc              = auc(fpr, tpr)
    roc_auc_curve        = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='AutoGluon')

    roc_auc_curve.plot()
    plt.show()