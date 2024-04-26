import numpy as np

np.seterr(all="ignore")


def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    x_exp = np.exp(x - np.max(x, axis=axis))
    return x_exp / np.sum(x_exp, axis=axis)
