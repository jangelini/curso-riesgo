from argparse import ArgumentParser

import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier

from optuna import create_study
from optuna.samplers import TPESampler
from optuna import Trial
from sklearn.impute import SimpleImputer

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer

from category_encoders import CatBoostEncoder
from sklearn.pipeline import Pipeline

from optimization_functions import instantiate_catencoder, instantiate_imputer, instantiate_lgbm, ks_score

# Esto permite pasarle argumentos cuando corremos la función en la terminal
parser = ArgumentParser(description='Read, process and save the datasets.')
#parser.add_argument('--data', type=str, help='The path of the directory containing the data.')
#parser.add_argument('--index', help='The name of the index column.')
args = parser.parse_args()

def optimize(trial : Trial, x, y, numerical_features, categorical_features) -> float:

    encoder : CatBoostEncoder = instantiate_catencoder(trial)
    imputer : SimpleImputer   = instantiate_imputer(trial)
    model   : LGBMClassifier  = instantiate_lgbm(trial)
    
    folds : StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    preprocessor = ColumnTransformer([
        ('imputer', imputer, numerical_features)
        ('categorical_encoder', encoder, categorical_features)
    ])

    pipeline : Pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    scores : float = cross_val_score(pipeline, x, y, scoring=ks_score, cv=folds)
    return np.mean(scores)