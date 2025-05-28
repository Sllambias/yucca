import torch


def torch_gamma(
    image: torch.Tensor,
    gamma_range=(0.5, 2),
    invert_image=False,
    epsilon=1e-7,
    per_channel=False,
    clip_to_input_range=False,
) -> torch.Tensor:
    if invert_image:
        image = -image

    if not per_channel:
        if torch.rand(1).item() < 0.5 and gamma_range[0] < 1:
            gamma = torch.rand(1).item() * (1 - gamma_range[0]) + gamma_range[0]
        else:
            gamma = torch.rand(1).item() * (gamma_range[1] - max(gamma_range[0], 1)) + max(gamma_range[0], 1)

        img_min = image.min()
        img_max = image.max()
        img_range = img_max - img_min

        image = torch.pow(((image - img_min) / (img_range + epsilon)), gamma) * img_range + img_min

        if clip_to_input_range:
            image = torch.clamp(image, min=img_min, max=img_max)
    else:
        for c in range(image.shape[0]):
            if torch.rand(1).item() < 0.5 and gamma_range[0] < 1:
                gamma = torch.rand(1).item() * (1 - gamma_range[0]) + gamma_range[0]
            else:
                gamma = torch.rand(1).item() * (gamma_range[1] - max(gamma_range[0], 1)) + max(gamma_range[0], 1)

            img_min = image[c].min()
            img_max = image[c].max()
            img_range = img_max - img_min

            image[c] = torch.pow(((image[c] - img_min) / (img_range + epsilon)), gamma) * (img_range + epsilon) + img_min

            if clip_to_input_range:
                image[c] = torch.clamp(image[c], min=img_min, max=img_max)

    return image
