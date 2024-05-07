import numpy as np
from scipy.ndimage import gaussian_filter


def blur(image, sigma, clip_to_input_range):
    img_min = image.min()
    img_max = image.max()

    image = gaussian_filter(image, sigma, order=0)
    if clip_to_input_range:
        image = np.clip(image, a_min=img_min, a_max=img_max)

    return image
