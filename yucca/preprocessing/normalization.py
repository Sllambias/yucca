import warnings
from skimage import exposure
import numpy as np


def normalizer(array: np.ndarray, scheme: str, intensities: {}):
    """
    Normalizing function for preprocessing and inference.

    supported schemes can be either:
    None = for no normalization. Generally not recommended.
    MinMax = for 0-1 or Min-Max normalization.
    Standardize = (array - mean) / std. Based on modality wide stats.
    Clip = for contrast clipping. This will clip values to the 0.01 and 99.99th percentiles
        and then perform 0-1 normalization.
    """
    accepted_schemes = ["clipping", "minmax", "no_norm", "standardize"]

    assert scheme in accepted_schemes, "invalid normalization scheme inserted" f"attempted scheme: {scheme}"

    if scheme == "no_norm":
        return array

    if scheme == "minmax":
        raise NotImplementedError("Min Max normalization is not implemented yet. Use standardize, clip or no_norm.")

    if scheme == "standardize":
        assert intensities is not None, "ERROR: dataset wide stats are required for standardize"
        return (array - float(intensities["mean"])) / float(intensities["std"])

    if scheme == "clip":
        lower_bound, upper_bound = np.percentile(array, (0.01, 99.99))
        array = exposure.rescale_intensity(array, in_range=(lower_bound, upper_bound), out_range=(0, 1))
        return array

    if scheme == "volume_wise_znorm":
        empty_val = np.min(array)  # We assume the background is the minimum value

        if empty_val != array.item(0):
            warnings.warn(
                "Tried to normalize an array where the top right value was not the same as the minimum value."
                f"empty_val: {empty_val}, top right: {array.item(0)}"
            )

        array = clamp(array)
        array = znormalize(array, array == empty_val)
        array = rescale(array, range(-1, 1))
        return array


def clamp(x, q=0.99):
    q_val = np.percentile(x, q)
    return np.clip(x, a_min=None, a_max=q_val)


def znormalize(x, mask):
    values = x[mask]
    mean, std = np.mean(values), np.std(values)
    if std == 0:
        return None
    x -= mean
    x /= std
    return x


def rescale(x, range=(0, 1)):
    return exposure.rescale_intensity(x, out_range=range)
