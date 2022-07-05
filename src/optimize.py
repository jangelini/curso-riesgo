from argparse import ArgumentParser

import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from optuna import create_study
from optuna.samplers import TPESampler

# Esto permite pasarle argumentos cuando corremos la función en la terminal
parser = ArgumentParser(description='Read, process and save the datasets.')
#parser.add_argument('--data', type=str, help='The path of the directory containing the data.')
#parser.add_argument('--index', help='The name of the index column.')
args = parser.parse_args()

