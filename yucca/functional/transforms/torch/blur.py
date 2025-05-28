import torch
import torchvision.transforms.functional as TF


def torch_blur(image: torch.tensor, sigma: float, clip_to_input_range: bool = True) -> torch.tensor:
    img_min = image.min()
    img_max = image.max()

    if image.ndim == 2:
        image = blur_2D(image, sigma)
    elif image.ndim == 3:
        image = blur_3D(image, sigma)
    else:
        raise ValueError(f"Unsupported image shape for blur: {image.shape}")

    if clip_to_input_range:
        image = torch.clamp(image, min=img_min, max=img_max)
    return image


def blur_2D(image: torch.tensor, sigma: float) -> torch.tensor:
    assert image.ndim == 2, "Expected [H, W] tensor"

    image = image.unsqueeze(0).unsqueeze(0)
    image = TF.gaussian_blur(image, kernel_size=int(2 * round(3 * sigma) + 1), sigma=sigma)
    return image.squeeze(0).squeeze(0)


def blur_3D(image: torch.Tensor, sigma: float) -> torch.Tensor:
    assert image.ndim == 3, "Expected [D, H, W] tensor"

    kernel_size = int(2 * round(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # : filter single-channel, [out_channels, in_channels, kernel_size]
    coords = torch.arange(kernel_size) - kernel_size // 2
    kernel = torch.exp(-(coords.float() ** 2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)

    # the permutation ordering is moving the target spatial dimension slice to the last dimension since
    # this is where conv1d applies the filter.
    volume = image
    for axis in range(3):
        permute_order = list(range(3))
        permute_order[axis], permute_order[-1] = permute_order[-1], permute_order[axis]
        volume = volume.permute(permute_order).contiguous()

        shape = volume.shape
        volume = volume.view(-1, 1, shape[-1])
        volume = torch.nn.functional.conv1d(volume, kernel.to(volume.device), padding=kernel.shape[-1] // 2)
        volume = volume.view(shape)
        volume = volume.permute(permute_order).contiguous()

    return volume
