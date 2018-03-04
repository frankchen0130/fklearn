# -*- coding:utf8 -*-
# Created by frank at 04/03/2018
import numpy as np

def log_loss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    _ = [i*np.log(j)+(1-i)*np.log(1-j) for i, j in zip(y_true, y_pred)]
    return -(sum(_))/ len(y_true)


