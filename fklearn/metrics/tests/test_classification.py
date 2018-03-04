from __future__ import division, print_function

import numpy as np
from scipy import linalg
from functools import partial
from itertools import product
import warnings

from sklearn import datasets
from sklearn import svm

from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import label_binarize
from sklearn.utils.validation import check_random_state

from sklearn.utils.testing import assert_raises, clean_warning_registry
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_warns
from sklearn.utils.testing import assert_no_warnings
from sklearn.utils.testing import assert_warns_message
from sklearn.utils.testing import assert_not_equal
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.mocking import MockDataFrame

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import hinge_loss
from sklearn.metrics import jaccard_similarity_score
from fklearn.metrics import log_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import brier_score_loss

from sklearn.metrics.classification import _check_targets
from sklearn.exceptions import UndefinedMetricWarning

from scipy.spatial.distance import hamming as sp_hamming

###############################################################################
# Utilities for testing



def test_log_loss():
    # binary case with symbolic labels ("no" < "yes")
    y_true = ["no", "no", "no", "yes", "yes", "yes"]
    y_pred = np.array([[0.5, 0.5], [0.1, 0.9], [0.01, 0.99],
                       [0.9, 0.1], [0.75, 0.25], [0.001, 0.999]])
    loss = log_loss(y_true, y_pred)
    assert_almost_equal(loss, 1.8817971)

    # multiclass case; adapted from http://bit.ly/RJJHWA
    y_true = [1, 0, 2]
    y_pred = [[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]]
    loss = log_loss(y_true, y_pred, normalize=True)
    assert_almost_equal(loss, 0.6904911)

    # check that we got all the shapes and axes right
    # by doubling the length of y_true and y_pred
    y_true *= 2
    y_pred *= 2
    loss = log_loss(y_true, y_pred, normalize=False)
    assert_almost_equal(loss, 0.6904911 * 6, decimal=6)

    # check eps and handling of absolute zero and one probabilities
    y_pred = np.asarray(y_pred) > .5
    loss = log_loss(y_true, y_pred, normalize=True, eps=.1)
    assert_almost_equal(loss, log_loss(y_true, np.clip(y_pred, .1, .9)))

    # raise error if number of classes are not equal.
    y_true = [1, 0, 2]
    y_pred = [[0.2, 0.7], [0.6, 0.5], [0.4, 0.1]]
    assert_raises(ValueError, log_loss, y_true, y_pred)

    # case when y_true is a string array object
    y_true = ["ham", "spam", "spam", "ham"]
    y_pred = [[0.2, 0.7], [0.6, 0.5], [0.4, 0.1], [0.7, 0.2]]
    loss = log_loss(y_true, y_pred)
    assert_almost_equal(loss, 1.0383217, decimal=6)

    # test labels option

    y_true = [2, 2]
    y_pred = [[0.2, 0.7], [0.6, 0.5]]
    y_score = np.array([[0.1, 0.9], [0.1, 0.9]])
    error_str = ('y_true contains only one label (2). Please provide '
                 'the true labels explicitly through the labels argument.')
    assert_raise_message(ValueError, error_str, log_loss, y_true, y_pred)

    y_pred = [[0.2, 0.7], [0.6, 0.5], [0.2, 0.3]]
    error_str = ('Found input variables with inconsistent numbers of samples: '
                 '[3, 2]')
    assert_raise_message(ValueError, error_str, log_loss, y_true, y_pred)

    # works when the labels argument is used

    true_log_loss = -np.mean(np.log(y_score[:, 1]))
    calculated_log_loss = log_loss(y_true, y_score, labels=[1, 2])
    assert_almost_equal(calculated_log_loss, true_log_loss)

    # ensure labels work when len(np.unique(y_true)) != y_pred.shape[1]
    y_true = [1, 2, 2]
    y_score2 = [[0.2, 0.7, 0.3], [0.6, 0.5, 0.3], [0.3, 0.9, 0.1]]
    loss = log_loss(y_true, y_score2, labels=[1, 2, 3])
    assert_almost_equal(loss, 1.0630345, decimal=6)


def test_log_loss_pandas_input():
    # case when input is a pandas series and dataframe gh-5715
    y_tr = np.array(["ham", "spam", "spam", "ham"])
    y_pr = np.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1], [0.7, 0.2]])
    types = [(MockDataFrame, MockDataFrame)]
    try:
        from pandas import Series, DataFrame
        types.append((Series, DataFrame))
    except ImportError:
        pass
    for TrueInputType, PredInputType in types:
        # y_pred dataframe, y_true series
        y_true, y_pred = TrueInputType(y_tr), PredInputType(y_pr)
        loss = log_loss(y_true, y_pred)
        assert_almost_equal(loss, 1.0383217, decimal=6)

