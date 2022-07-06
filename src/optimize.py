from argparse import ArgumentParser
from typing import List
import os

import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier

from optuna import create_study, load_study
from optuna.samplers import TPESampler
from optuna import Trial

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer

from category_encoders import CatBoostEncoder
from sklearn.pipeline import Pipeline

from optimization_functions import instantiate_catencoder, instantiate_imputer, instantiate_lgbm, ks_score

# Esto permite pasarle argumentos cuando corremos la función en la terminal
# parser = ArgumentParser(description='Read, process and save the datasets.')
#parser.add_argument('--data', type=str, help='The path of the directory containing the data.')
#parser.add_argument('--index', help='The name of the index column.')
# args = parser.parse_args()

def objective(trial : Trial, x, y, numerical_features : List[str], categorical_features : List[str]) -> float:

    encoder : CatBoostEncoder = instantiate_catencoder(trial)
    imputer : SimpleImputer   = instantiate_imputer(trial)
    model   : LGBMClassifier  = instantiate_lgbm(trial)
    
    folds : StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    preprocessor = ColumnTransformer([
        ('imputer', imputer, numerical_features),
        ('categorical_encoder', encoder, categorical_features)
    ])

    pipeline : Pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    scores : float = cross_val_score(pipeline, x, y, scoring=ks_score, cv=folds)
    return np.mean(scores)

if __name__=='__main__':

    data    = pd.read_parquet('../data/full_train_pandas.parquet')

    index  : List[str] = ['SK_ID_CURR']
    target : List[str] = ['TARGET']

    categorical_columns : List[str] = [column for column, dtype in data.dtypes.to_dict().items() if dtype==np.object_ and column not in index+target]
    numerical_columns   : List[str] = [column for column in data.columns if column not in index+target+categorical_columns]

    x_train, x_test, y_train, y_test = train_test_split(data.drop(index+target, axis=1), data.TARGET, random_state=42)

    sampler = TPESampler(n_startup_trials=200, seed=42)

    if os.path.isfile('optimization.sqlite'):
        study = load_study(study_name='optimizacion_riesgo', storage='sqlite:///optimization.sqlite', sampler=sampler)
    else:
        study = create_study(storage='sqlite:///optimization.sqlite', direction='maximize', study_name='optimizacion_riesgo')

    study.optimize(lambda trial: objective(trial, x_train, y_train, numerical_columns, categorical_columns),
                   n_trials=3000
    )