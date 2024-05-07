import numpy as np


def mask_batch(batch, pixel_value, ratio, token_size):
    assert len(batch.shape[2:]) == len(token_size), (
        "mask token size not compatible with input data" f"mask token is: {token_size} and image is shape: {batch.shape[2:]}"
    )
    # np.ceil to get a grid with exact or larger dims than the input image
    # later we will crop it to the desired dims
    slices = [slice(0, shape) for shape in batch.shape[2:]]
    grid_dims = np.ceil(batch.shape[2:] / np.array(token_size)).astype(np.uint8)

    grid_flat = np.ones(np.prod(grid_dims))
    grid_flat[: int(len(grid_flat) * ratio)] = 0
    np.random.shuffle(grid_flat)
    grid = grid_flat.reshape(grid_dims)
    for idx, size in enumerate(token_size):
        grid = np.repeat(grid, repeats=size, axis=idx)

    batch[:, :, grid[*slices] == 0] = pixel_value
    return batch
