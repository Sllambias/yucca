from skimage import exposure
import numpy as np


def normalizer(array, scheme, intensities):
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
