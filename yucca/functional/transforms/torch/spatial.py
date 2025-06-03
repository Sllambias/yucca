import torch
import torch.nn.functional as F
from typing import Optional, Union
from yucca.functional.array_operations.matrix_ops import Rx, Ry, Rz, Rz2D


def _create_zero_centered_coordinate_matrix(shape: tuple[int, ...]) -> torch.Tensor:
    ranges = [torch.arange(s, dtype=torch.float32) for s in shape]
    mesh = torch.stack(torch.meshgrid(*ranges, indexing="ij"))  # (D, ...)
    for d, s in enumerate(shape):
        mesh[d] -= (s - 1) / 2.0
    return mesh

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
    interpolation_mode: str = "bilinear",
    cval: Optional[Union[str, float]] = "min",
    seed: Optional[int] = None,
):
    if seed is not None:
        torch.manual_seed(seed)

    device, dtype = image.device, image.dtype
    ndim = len(patch_size)

    # Expand dims if needed
    def _prepare(x): 
        return x[None, None] if x.ndim == ndim else x[None] if x.ndim == ndim + 1 else x
    image = _prepare(image)
    if label is not None:
        label = _prepare(label)

    if not do_crop:
        patch_size = image.shape[2:]

    coords = _create_zero_centered_coordinate_matrix(patch_size).to(device, dtype)

    if torch.rand(1) < p_deform:
        noise = torch.randn(1, ndim, *patch_size, device=device, dtype=dtype)
        if ndim == 2:
            # Separable 2D blur
            ksize = 21
            ax = torch.arange(ksize, device=device, dtype=dtype) - ksize // 2
            k = torch.exp(-0.5 * (ax / sigma) ** 2)
            k /= k.sum()
            ky = k.view(1, 1, -1, 1)
            kx = k.view(1, 1, 1, -1)
            noise = F.conv2d(noise, ky, padding=(ksize//2, 0), groups=ndim)
            noise = F.conv2d(noise, kx, padding=(0, ksize//2), groups=ndim)
        else:
            # Separable 3D blur
            ksize = 9
            ax = torch.arange(ksize, device=device, dtype=dtype) - ksize // 2
            k = torch.exp(-0.5 * (ax / sigma) ** 2)
            k /= k.sum()
            for dim in range(3):
                shape = [1, 1, 1, 1, 1]
                shape[dim + 2] = ksize
                kernel = k.view(*shape).repeat(ndim, 1, 1, 1, 1)
                padding = [ksize // 2 if i == dim else 0 for i in range(3)]
                noise = F.conv3d(noise, kernel, padding=padding, groups=ndim)
        coords += (noise[0] * alpha)

    if torch.rand(1) < p_rot:
        rot = torch.eye(ndim, device=device, dtype=dtype)
        if ndim == 2:
            rot = rot @ torch.from_numpy(Rz2D(z_rot)).to(device=device, dtype=dtype)
        else:
            if torch.rand(1) < p_rot_per_axis:
                rot = rot @ torch.from_numpy(Rx(x_rot)).to(device=device, dtype=dtype)
            if torch.rand(1) < p_rot_per_axis:
                rot = rot @ torch.from_numpy(Ry(y_rot)).to(device=device, dtype=dtype)
            if torch.rand(1) < p_rot_per_axis:
                rot = rot @ torch.from_numpy(Rz(z_rot)).to(device=device, dtype=dtype)
        coords = (coords.view(ndim, -1).T @ rot).T.view_as(coords)

    if torch.rand(1) < p_scale:
        coords *= scale_factor

    if random_crop and do_crop:
        for d in range(ndim):
            lo = patch_size[d] // 2
            hi = image.shape[d + 2] - patch_size[d] // 2 + 1
            coords[d] += torch.randint(lo, hi, (1,), device=device)
    else:
        for d in range(ndim):
            coords[d] += image.shape[d + 2] / 2 - 0.5

    for d in range(ndim):
        coords[d] = 2 * coords[d] / (image.shape[d + 2] - 1) - 1

    grid = coords.permute(*range(1, ndim + 1), 0)[None]
    grid_sample_args = {
        "mode": interpolation_mode,
        "padding_mode": "zeros",
        "align_corners": True
    }

    image_canvas = F.grid_sample(image, grid, **grid_sample_args)
    if clip_to_input_range:
        image_canvas = torch.clamp(image_canvas, min=image.min(), max=image.max())

    if label is not None and not skip_label:
        label_canvas = F.grid_sample(label.float(), grid, mode="nearest", padding_mode="zeros", align_corners=True)
        label_canvas = label_canvas.to(label.dtype)
    else:
        label_canvas = None

    def _restore(x: Optional[torch.Tensor], target_ndim: int) -> Optional[torch.Tensor]:
        if x is None:
            return None
        # For 3D volumes, we want to keep the spatial dimensions
        if target_ndim == 3:
            return x.squeeze(0) if x.ndim == 4 else x
        # For 2D slices, remove all extra dimensions
        return x.squeeze()
    
    return _restore(image_canvas, image.ndim), _restore(label_canvas, label.ndim if label is not None else 0) if label_canvas is not None else None
