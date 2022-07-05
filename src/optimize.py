from argparse import ArgumentParser

import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from optuna import create_study
from optuna.samplers import TPESampler
from optuna import Trial

# Esto permite pasarle argumentos cuando corremos la función en la terminal
parser = ArgumentParser(description='Read, process and save the datasets.')
#parser.add_argument('--data', type=str, help='The path of the directory containing the data.')
#parser.add_argument('--index', help='The name of the index column.')
args = parser.parse_args()

def optimize(trial : Trial, x, y):
    boosting_type = trial.suggest_categorical('boosting_type', ['rf', 'gbdt', 'dart', 'goss'])
    n_estimators = trial.suggest_int('n_estimators', 5, 3000)
    max_depth = trial.suggest_int('max_depth', 2, 300)
    num_leaves = trial.suggest_int('num_leaves', 2, 2**max_depth - 1)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-7, 0.01)
    min_split_gain = trial.suggest_float('min_split_gain', 0, 10)
    subsample = trial.suggest_float('subsample', 0, 1)
    subsample_freq = trial.suggest_int('subsample_freq', 1, n_estimators)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.01, 1)
    reg_alpha = trial.suggest_float('reg_alpha', 0, 100)
    reg_lambda = trial.suggest_float('reg_lambda', 0, 100)
    random_state = 42

    lgbm = LGBMClassifier(boosting_type, 
                          num_leaves,
                          max_depth,
                          learning_rate,
                          n_estimators,
                          min_split_gain=min_split_gain,
. )
