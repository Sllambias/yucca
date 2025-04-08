from skimage import exposure
import numpy as np
from typing import Optional


def normalizer(array: np.ndarray, scheme: str, intensities: Optional[dict] = None):
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
    accepted_schemes = ["clipping", "ct", "minmax", "range", "no_norm", "standardize", "volume_wise_znorm", "255to1"]

    assert scheme in accepted_schemes, "invalid normalization scheme inserted" f"attempted scheme: {scheme}"
    assert array is not None

    if scheme == "no_norm":
        return array

    elif scheme == "minmax":
        assert intensities is not None, "ERROR: dataset wide stats are required for minmax"
        return (array - intensities["min"]) / (intensities["max"] - intensities["min"] + 1e-9)

    elif scheme == "255to1":
        return array / 255

    elif scheme == "range":
        print(array.max(), array.min(), intensities)
        return (array - array.min()) / (array.max() - array.min() + 1e-9) * (
            intensities["max"] - intensities["min"] + 1e-9
        ) + intensities["min"]

    elif scheme == "standardize":
        assert intensities is not None, "ERROR: dataset wide stats are required for standardize"
        return (array - float(intensities["mean"])) / float(intensities["std"])

    elif scheme == "clip":
        lower_bound, upper_bound = np.percentile(array, (0.01, 99.99))
        array = exposure.rescale_intensity(array, in_range=(lower_bound, upper_bound), out_range=(0, 1))
        return array

    elif scheme == "volume_wise_znorm":
        empty_val = array.min()  # We assume the background is the minimum value
        mask = array != empty_val
        array = clamp(array, mask=mask)
        array = znormalize(array, mask=mask)
        array = rescale(array, range=(0, 1))
        return array

    elif scheme == "ct":
        mean_intensity = float(intensities["mean"])
        std_intensity = float(intensities["std"])
        lower_bound = float(intensities["percentile_00_5"])
        upper_bound = float(intensities["percentile_99_5"])

        array = array.astype(np.float32, copy=False)
        np.clip(array, lower_bound, upper_bound, out=array)
        array -= mean_intensity
        array /= max(std_intensity, 1e-8)
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
