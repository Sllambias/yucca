import torch


def torch_gibbs_ringing(
    image: torch.Tensor, num_sample: int, mode: str = "rect", axes: list[int] = None, clip_to_input_range: bool = False
) -> torch.Tensor:
    assert image.ndim in [2, 3], "Only 2D or 3D images supported"
    if mode == "rect":
        assert axes is not None and all(0 <= ax < image.ndim for ax in axes), f"Invalid axes for mode 'rect'"

    img_min = image.min()
    img_max = image.max()
    offset = -img_min if img_min < 0 else 0
    image = image + offset

    kspace = torch.fft.fftshift(torch.fft.fftn(image, dim=list(range(image.ndim))), dim=list(range(image.ndim)))

    shape = image.shape
    center = [s // 2 for s in shape]

    if mode == "rect":
        mask = torch.ones_like(kspace, dtype=torch.bool)
        for axis in axes:
            c = center[axis]
            half = num_sample // 2
            slc = [slice(None)] * image.ndim
            for i in range(shape[axis]):
                if not (c - half <= i < c + half):
                    slc[axis] = slice(i, i + 1)
                    mask[tuple(slc)] = False
        kspace[~mask] = 0
    elif mode == "radial":
        coords = torch.meshgrid([torch.arange(s, device=image.device) - c for s, c in zip(shape, center)], indexing="ij")
        dist = torch.sqrt(sum((g.float() ** 2 for g in coords)))
        mask = dist <= num_sample
        kspace[~mask] = 0
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    result = torch.fft.ifftn(torch.fft.ifftshift(kspace, dim=list(range(image.ndim))), dim=list(range(image.ndim)))
    result = result.abs()

    result = result - offset
    if clip_to_input_range:
        result = torch.clamp(result, min=img_min, max=img_max)

    return result
