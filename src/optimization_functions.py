import numpy as np

from scipy.stats import ks_2samp

def ks(y_true, y_pred):
    class0 = y_pred[y_true == 0]
    class1 = y_pred[y_true == 1]
    result = ks_2samp(class0, class1).statistic
    return result


