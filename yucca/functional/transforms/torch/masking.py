import torch


def torch_mask(image: torch.Tensor, pixel_value: float, ratio: float, token_size: list[int]) -> torch.Tensor:
    """
    We need to mask image over all channels thus input should be 4d tensor of shape (c, x, y, z) or 3d tensor of shape (c, x, y)
    """

    input_shape = image.shape[1:]  # spatial dims

    if len(token_size) == 1:
        token_size *= len(input_shape)
    assert len(input_shape) == len(
        token_size
    ), f"mask token size not compatible with input data — token: {token_size}, image shape: {input_shape}"

    input_shape_tensor = image.new_tensor(input_shape, dtype=torch.int)
    token_size_tensor = image.new_tensor(token_size, dtype=torch.int)
    grid_dims = torch.ceil(input_shape_tensor / token_size_tensor).to(dtype=torch.int)
    grid_size = torch.prod(grid_dims).item()

    grid_flat = image.new_ones(grid_size)

    grid_flat[: int(grid_size * ratio)] = 0
    grid_flat = grid_flat[torch.randperm(grid_size, device=image.device)]

    grid = grid_flat.view(*grid_dims)

    for dim, size in enumerate(token_size):
        grid = grid.repeat_interleave(size, dim=dim)

    slices = tuple(slice(0, s) for s in input_shape)
    mask = grid[slices]

    image[:, mask == 0] = pixel_value
    return image, mask
