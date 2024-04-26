import numpy as np


def additive_noise(image, mean, sigma, clip_to_input_range: bool = False):
    # J = I+n
    img_min = image.min()
    img_max = image.max()
    image += np.random.normal(mean, sigma, image.shape)
    if clip_to_input_range:
        image = np.clip(image, a_min=img_min, a_max=img_max)
    return image


def multiplicative_noise(image, mean, sigma, clip_to_input_range: bool = False):
    # J = I + I*n
    img_min = image.min()
    img_max = image.max()
    gauss = np.random.normal(mean, sigma, image.shape)
    image += image * gauss
    if clip_to_input_range:
        image = np.clip(image, a_min=img_min, a_max=img_max)
    return image
