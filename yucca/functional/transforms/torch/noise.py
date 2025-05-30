import torch


def torch_additive_noise(image, mean, sigma, clip_to_input_range: bool = False):
    # J = I+n
    img_min = image.min()
    img_max = image.max()
    image = image + torch.normal(mean, sigma, image.shape, device=image.device)
    if clip_to_input_range:
        image = torch.clamp(image, min=img_min, max=img_max)
    return image


def torch_multiplicative_noise(image, mean, sigma, clip_to_input_range: bool = False):
    # J = I + I*n
    img_min = image.min()
    img_max = image.max()
    gauss = torch.normal(mean, sigma, image.shape, device=image.device)
    image = image + image * gauss
    if clip_to_input_range:
        image = torch.clamp(image, min=img_min, max=img_max)
    return image
