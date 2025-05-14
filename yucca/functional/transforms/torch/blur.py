import torch
import torchvision.transforms.functional as TF

def torch_blur(
        image: torch.tensor,
        input_dims: torch.tensor,
        sigma: float,
        clip_to_input_range: bool = True
    ):
    img_min = image.min()
    img_max = image.max()

    if input_dims == 2:
        image = blur_2D_case_from_2D(image, sigma)
    elif input_dims == 3 and image.shape[0] <= 3:
        image = blur_2D_case_from_3D(image, sigma)
    elif input_dims == 3:
        image = blur_3D_case_from_3D(image, sigma)
    else:
        raise ValueError(f"Unsupported image shape for blur: {image.shape}")

    if clip_to_input_range:
        image = torch.clamp(image, min=img_min, max=img_max)
    return image.cpu().numpy() if not isinstance(image, torch.Tensor) else image

def blur_2D_case_from_2D(image: torch.tensor, sigma: float) -> torch.tensor:
    image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    image = TF.gaussian_blur(image, kernel_size=int(2 * round(3 * sigma) + 1), sigma=sigma)
    return image.squeeze(0).squeeze(0)

def blur_2D_case_from_3D(image: torch.tensor, sigma: float) -> torch.tensor:
    image = image.unsqueeze(0)  # (1, C, H, W)
    image = TF.gaussian_blur(image, kernel_size=int(2 * round(3 * sigma) + 1), sigma=sigma)
    return image.squeeze(0)

def blur_3D_case_from_3D(image: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel_size = int(2 * round(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    coords = torch.arange(kernel_size, device=image.device) - kernel_size // 2
    kernel = torch.exp(-(coords.float() ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1) # single-channel filter

    # We are dealing with a voxel volume here but we'd like to support batches of 
    # volumes and multi-modal MRI so adding batch and channel dimensions should support that.
    image = image.unsqueeze(0).unsqueeze(0)

    # the permutation ordering is moving the target spatial dimension slice to the last dimension since 
    # this is where conv1d applies the filter.
    for axis in range(2, 5):
        permute_order = list(range(image.dim()))
        permute_order[axis], permute_order[-1] = permute_order[-1], permute_order[axis]
        image = image.permute(permute_order).contiguous()
        orig_shape = image.shape

        image = image.reshape(-1, 1, orig_shape[-1])
        image = torch.nn.functional.conv1d(image, kernel, padding=kernel.shape[-1] // 2)

        image = image.reshape(orig_shape)
        image = image.permute(permute_order).contiguous()
    image = image.squeeze(0).squeeze(0)
    if image.shape[0] != image.shape[1]:
        image = image.permute(1, 2, 0)
    return image
