import warnings
from skimage import exposure
import numpy as np


def normalizer(array: np.ndarray, scheme: str, intensities: {}):
    """
    Normalizing function for preprocessing and inference.

    supported schemes can be either:
    None = for no normalization. Generally not recommended.
    MinMax = for 0-1 or Min-Max normalization.
    255to1 = for 0-1 normalization of 8 bit images.
    Standardize = (array - mean) / std. Based on modality wide stats.
    Clip = for contrast clipping. This will clip values to the 0.01 and 99.99th percentiles
        and then perform 0-1 normalization.
    """
    accepted_schemes = ["clipping", "minmax", "no_norm", "standardize", "volume_wise_znorm", "255to1"]

    assert scheme in accepted_schemes, "invalid normalization scheme inserted" f"attempted scheme: {scheme}"
    assert array is not None

    if scheme == "no_norm":
        return array

    elif scheme == "minmax":
        assert intensities is not None, "ERROR: dataset wide stats are required for minmax"
        return (array - intensities["min"]) / (intensities["max"] - intensities["min"])

    elif scheme == "255to1":
        return array / 255

    elif scheme == "standardize":
        assert intensities is not None, "ERROR: dataset wide stats are required for standardize"
        return (array - float(intensities["mean"])) / float(intensities["std"])

    elif scheme == "clip":
        lower_bound, upper_bound = np.percentile(array, (0.01, 99.99))
        array = exposure.rescale_intensity(array, in_range=(lower_bound, upper_bound), out_range=(0, 1))
        return array

    elif scheme == "volume_wise_znorm":
        empty_val = array.min()  # We assume the background is the minimum value

        if empty_val != array[0, 0, 0]:
            warnings.warn(
                "Tried to normalize an array where the top right value was not the same as the minimum value."
                f"empty_val: {empty_val}, top right: {array[0, 0, 0]}"
            )
        mask = array != empty_val
        array = clamp(array, mask=mask)
        array = znormalize(array, mask=mask)
        array = rescale(array, range=(0, 1))
        return array


def clamp(x, mask, q=0.99):
    q_val = np.quantile(x[mask], q)
    return np.clip(x, a_min=None, a_max=q_val)


def znormalize(x, mask):
    values = x[mask]
    mean, std = np.mean(values), np.std(values)
    assert std > 0
    x -= mean
    x /= std
    return x


def rescale(x, range=(0, 1)):
    return exposure.rescale_intensity(x, out_range=range)
