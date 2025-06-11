import torch
import torch.nn.functional as F


def torch_simulate_lowres(
    image: torch.Tensor,
    target_shape: tuple[int],
    clip_to_input_range: bool,
) -> torch.Tensor:
    original_shape = image.shape
    img_min = image.min()
    img_max = image.max()

    image = image.float()
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
        mode_down = "nearest-exact"
        mode_up = "bicubic"
    elif image.ndim == 3:
        image = image.unsqueeze(0).unsqueeze(0)
        mode_down = "nearest-exact"
        mode_up = "trilinear"
    else:
        raise ValueError("Image must be 3D or 4D.")

    downsampled = F.interpolate(image, size=tuple(target_shape), mode=mode_down)
    upsampled = F.interpolate(downsampled, size=tuple(original_shape), mode=mode_up, align_corners=False)
    result = upsampled.squeeze(0).squeeze(0)

    if clip_to_input_range:
        result = result.clamp(min=img_min.item(), max=img_max.item())

    return result
