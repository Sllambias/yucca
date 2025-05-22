import torch

def torch_mask(
        image: torch.Tensor, 
        pixel_value: float, 
        ratio: float, 
        token_size: tuple[int]
    ) -> torch.Tensor:

    if image.ndim == 2:
        image = image.unsqueeze(0)
    elif image.ndim == 3:
        image = image.unsqueeze(0)

    input_shape = image.shape[1:]  # spatial dims
    assert len(input_shape) == len(token_size), (
        f"mask token size not compatible with input data — token: {token_size}, image shape: {input_shape}"
    )

    input_shape_tensor = image.new_tensor(input_shape, dtype=torch.int)
    token_size_tensor = image.new_tensor(token_size, dtype=torch.int)

    grid_dims = torch.ceil(input_shape_tensor / token_size_tensor).to(dtype=torch.int)
    grid_size = torch.prod(grid_dims).item()

    grid_flat = image.new_ones(grid_size)
    grid_flat[:int(grid_size * ratio)] = 0
    grid_flat = grid_flat[torch.randperm(grid_size, device=image.device)]

    grid = grid_flat.view(*grid_dims)

    for dim, size in enumerate(token_size):
        grid = grid.repeat_interleave(size, dim=dim)

    slices = tuple(slice(0, s) for s in input_shape)
    mask = grid[slices]

    image[:, mask == 0] = pixel_value

    return image.squeeze(0)