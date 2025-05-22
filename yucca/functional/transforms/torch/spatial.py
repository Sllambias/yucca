import torch
import torch.nn.functional as F
from typing import Optional, Union
from yucca.functional.array_operations.matrix_ops import (
    create_zero_centered_coordinate_matrix,
    deform_coordinate_matrix,
    Rx, Ry, Rz, Rz2D,
)


def torch_spatial(
    image: torch.Tensor,
    patch_size: tuple[int],
    p_deform: float,
    p_rot: float,
    p_rot_per_axis: float,
    p_scale: float,
    alpha: float,
    sigma: float,
    x_rot: float,
    y_rot: float,
    z_rot: float,
    scale_factor: float,
    clip_to_input_range: bool,
    label: Optional[torch.Tensor] = None,
    skip_label: bool = False,
    do_crop: bool = True,
    random_crop: bool = True,
    interpolation_mode: str = "bilinear",  # was: order
    cval: Optional[Union[str, float]] = "min",
    seed: Optional[int] = None,
):
    device = image.device
    dtype = image.dtype
    ndim = len(patch_size)

    if seed is not None:
        torch.manual_seed(seed)

    # Store original ndim for shape restoration
    original_image_ndim = image.ndim
    original_label_ndim = label.ndim if label is not None else None

    # : handle standard torch image dimensions
    if image.dim() == ndim:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == ndim + 1:
        image = image.unsqueeze(0)
    elif image.dim() != ndim + 2:
        raise ValueError(f"Expected input to have {ndim} spatial dimensions, got {image.dim()} dimensions")

    if label is not None:
        if label.dim() == ndim:
            label = label.unsqueeze(0).unsqueeze(0)
        elif label.dim() == ndim + 1:
            label = label.unsqueeze(0)
        elif label.dim() != ndim + 2:
            raise ValueError(f"Expected label to have {ndim} spatial dimensions, got {label.dim()} dimensions")

    # : cropping, use full image size
    if not do_crop:
        patch_size = image.shape[2:]

    # : clipping stats
    if cval == "min":
        cval = float(image.min())
    else:
        assert isinstance(cval, (int, float)), f"Invalid cval: {cval}"

    img_min = image.min()
    img_max = image.max()

    # : deformation
    coords_np = create_zero_centered_coordinate_matrix(patch_size)
    if torch.rand(1) < p_deform:
        coords_np = deform_coordinate_matrix(coords_np, alpha=alpha, sigma=sigma)
    coords = torch.from_numpy(coords_np).to(device=device, dtype=dtype)

    # : rotation
    if torch.rand(1) < p_rot:
        rot_matrix = torch.eye(ndim, device=device, dtype=dtype)
        if ndim == 2:
            rot_matrix = rot_matrix @ torch.from_numpy(Rz2D(z_rot)).to(device=device, dtype=dtype)
        else:
            if torch.rand(1) < p_rot_per_axis:
                rot_matrix = rot_matrix @ torch.from_numpy(Rx(x_rot)).to(device=device, dtype=dtype)
            if torch.rand(1) < p_rot_per_axis:
                rot_matrix = rot_matrix @ torch.from_numpy(Ry(y_rot)).to(device=device, dtype=dtype)
            if torch.rand(1) < p_rot_per_axis:
                rot_matrix = rot_matrix @ torch.from_numpy(Rz(z_rot)).to(device=device, dtype=dtype)
        coords = torch.matmul(coords.reshape(ndim, -1).T, rot_matrix).T.reshape(coords.shape)

    # : scaling
    if torch.rand(1) < p_scale:
        coords *= scale_factor

    # : cropping, random or centered
    if random_crop and do_crop:
        for d in range(ndim):
            crop_center = torch.randint(
                low=patch_size[d] // 2,
                high=image.shape[d + 2] - patch_size[d] // 2 + 1,
                size=(1,),
                device=device
            )
            coords[d] += crop_center
    else:
        for d in range(ndim):
            coords[d] += image.shape[d + 2] / 2.0 - 0.5

    # : normalize to [-1, 1] for grid_sample
    for d in range(ndim):
        coords[d] = 2.0 * coords[d] / (image.shape[d + 2] - 1) - 1.0

    if ndim == 2:
        grid = coords.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, 2)
    else:
        grid = coords.permute(1, 2, 3, 0).unsqueeze(0)  # (1, D, H, W, 3)

    if interpolation_mode not in {"bilinear", "nearest"}:
        raise ValueError("interpolation_mode must be 'bilinear' or 'nearest'")

    image_canvas = F.grid_sample(
        image,
        grid,
        mode=interpolation_mode,
        padding_mode='zeros',
        align_corners=True
    )

    if clip_to_input_range:
        image_canvas = torch.clamp(image_canvas, min=img_min, max=img_max)

    if label is not None and not skip_label:
        label_canvas = F.grid_sample(
            label,
            grid,
            mode='nearest',
            padding_mode='zeros',
            align_corners=True
        )

    def restore_shape(t: torch.Tensor, original_ndim: int) -> torch.Tensor:
        while t.ndim > original_ndim:
            t = t.squeeze(0)
        return t

    image_canvas = restore_shape(image_canvas, original_image_ndim)

    if label is not None and not skip_label:
        label_canvas = restore_shape(label_canvas, original_label_ndim)
        return image_canvas, label_canvas

    return image_canvas, None
