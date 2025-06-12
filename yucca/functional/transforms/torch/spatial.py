import torch
import torch.nn.functional as F
from typing import Optional
from yucca.functional.array_operations.matrix_ops import Rx, Ry, Rz, Rz2D


def _create_zero_centered_coordinate_matrix(shape: tuple[int, ...]) -> torch.Tensor:
    # Using standard numpy indexing 'ij', 2D: (H, W) = (y, x), 3D: (D, H, W) = (z, y, x)
    mesh = torch.stack(torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in shape], indexing="ij"))
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
    seed: Optional[int] = None,
):
    if seed is not None:
        torch.manual_seed(seed)

    device, dtype, orig_ndim = image.device, image.dtype, image.ndim
    ndim = len(patch_size)

    # Expand dims if needed
    def _prepare(x):
        return x[None, None] if orig_ndim == ndim else x[None] if orig_ndim == ndim + 1 else x

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
            noise = F.conv2d(noise, ky, padding=(ksize // 2, 0), groups=ndim)
            noise = F.conv2d(noise, kx, padding=(0, ksize // 2), groups=ndim)
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
        coords += noise[0] * alpha

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
        coords = (rot @ coords.view(ndim, -1)).view_as(coords)

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

    # Swap axes to (x, y) or (x, y, z) order for grid_sample (torch does not default to numpy indexing here)
    grid = coords.permute(*range(1, ndim + 1), 0)
    grid = torch.stack([grid] * image.shape[0], dim=0)

    if ndim == 2:
        grid = grid[..., [1, 0]]
    elif ndim == 3:
        grid = grid[..., [2, 1, 0]]
    else:
        raise ValueError("Only 2D and 3D supported")
    grid_sample_args = {"mode": interpolation_mode, "padding_mode": "zeros", "align_corners": True}
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
        current_dims = x.ndim
        for i in range(current_dims - target_ndim):
            x = x.squeeze(0)
        return x

    return _restore(image_canvas, orig_ndim), (
        _restore(label_canvas, orig_ndim if label is not None else 0) if label_canvas is not None else None
    )


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    array = torch.zeros((1, 12, 12, 12))
    array[:, 4:8, 4:8, 4:8] = 1
    array[:, 4, 4:8, 4:8] = 2
    array = torch.from_numpy(
        np.load("/Users/zcr545/Desktop/Projects/repos/asparagus_data/preprocessed_data/Task001_OASIS/imagesTr/1000.nii.npy")
    ).squeeze(0)
    array = torch.stack([array, array], dim=0)
    print(array.shape)
    array = torch.stack([array, array, array])
    print(array.shape)

    out, _ = torch_spatial(
        array,
        patch_size=array.shape[-3:],
        p_deform=0,
        p_rot=1,
        p_rot_per_axis=1,
        p_scale=0,
        alpha=5,  # 5-20
        sigma=10,  # 5-20
        x_rot=0,
        y_rot=0,
        z_rot=0.1,
        scale_factor=0,
        clip_to_input_range=False,
        do_crop=False,
    )
    print("IN", array.shape, "OUT", out.shape)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(array[0, :, 40], cmap="gray")
    ax[1].imshow(out[:, 40], cmap="gray")
