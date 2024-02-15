import numpy as np
import numpy.typing as npt
from typing import Literal
from sklearn.metrics import roc_auc_score


def dice(tp, fp, tn, fn):  # noqa: U100
    try:
        return (2 * tp) / (2 * tp + fp + fn)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def dice_per_label(tp_list, fp_list, tn_list, fn_list):
    return [dice(tp_list[i], fp_list[i], tn_list[i], fn_list[i]) for i in range(len(tp_list))]


def jaccard(tp, fp, tn, fn):  # noqa: U100
    try:
        return (tp) / (tp + fp + fn)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def jaccard_per_label(tp_list, fp_list, tn_list, fn_list):
    return [jaccard(tp_list[i], fp_list[i], tn_list[i], fn_list[i]) for i in range(len(tp_list))]


def sensitivity(tp, fp, tn, fn):  # noqa: U100
    # recall, hit rate, tpr
    # How many cases of X are correctly recognized as X?
    try:
        return tp / (tp + fn)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def specificity(tp, fp, tn, fn):  # noqa: U100
    # beware with using TN-metrics.
    try:
        return tn / (tn + fp)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def precision(tp, fp, tn, fn):  # noqa: U100
    # When X is predicted, how often is it truly X?
    try:
        return tp / (tp + fp)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def volume_similarity(tp, fp, tn, fn):  # noqa: U100
    try:
        return 1 - abs(fn - fp) / (2 * tp + fn + fp)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def f1(tp, fp, tn, fn):  # noqa: U100
    try:
        prec = precision(tp, fp, tn, fn)
        sens = sensitivity(tp, fp, tn, fn)
        return 2 * (prec * sens) / (prec + sens)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def accuracy(tp, fp, tn, fn):
    try:
        return (tp + tn) / (tp + fp + tn + fn)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def auroc(y_true: npt.ArrayLike, y_score: npt.ArrayLike, multi_class_mode: Literal["ovr", "ovo"] = "ovr"):
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC/AUROC) from prediction scores for each class.

    :param y_true: ground truth labels, shape (n_samples,n_classes) for multiclass or (n_samples,) for binary.
                    Both probabilities and labels are accepted.
    :param y_score: predicted scores, shape (n_samples,n_classes)
    :param multi_class_mode: In multiclass setting, calculate ROC AUC one-vs-one (ovo), or one-vs-rest (ovr). One of {'ovr', 'ovo'}, default='ovr'

    """
    assert len(y_true.shape) == 1 or y_true.shape == y_score.shape, "y_true must be 1D or 2D"
    return roc_auc_score(y_true, y_score, average=None, multi_class=multi_class_mode)


def TP(tp, fp, tn, fn):  # noqa: U100
    return tp


def FP(tp, fp, tn, fn):  # noqa: U100
    return fp


def FN(tp, fp, tn, fn):  # noqa: U100
    return fn


def total_pos_gt(tp, fp, tn, fn):  # noqa: U100
    return tp + fn


def total_pos_pred(tp, fp, tn, fn):  # noqa: U100
    return tp + fp
