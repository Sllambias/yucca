import torch


def torch_bias_field(image: torch.Tensor, clip_to_input_range: bool = False) -> torch.Tensor:
    device = image.device
    img_min = image.min()
    img_max = image.max()

    if len(image.shape) == 3:
        assert image.ndim == 3, "Expected [H, W, D] tensor"
        x, y, z = image.shape
        X, Y, Z = torch.meshgrid(
            torch.linspace(0, x - 1, x, device=device),
            torch.linspace(0, y - 1, y, device=device),
            torch.linspace(0, z - 1, z, device=device),
            indexing="ij",
        )
        x0 = torch.randint(0, x, (1,), device=device)
        y0 = torch.randint(0, y, (1,), device=device)
        z0 = torch.randint(0, z, (1,), device=device)
        G = 1 - ((X - x0) ** 2 / (x**2) + (Y - y0) ** 2 / (y**2) + (Z - z0) ** 2 / (z**2))
    else:
        assert image.ndim == 2, "Expected [H, W] tensor"

        x, y = image.shape
        X, Y = torch.meshgrid(
            torch.linspace(0, x - 1, x, device=device), torch.linspace(0, y - 1, y, device=device), indexing="ij"
        )
        x0 = torch.randint(0, x, (1,), device=device)
        y0 = torch.randint(0, y, (1,), device=device)
        G = 1 - ((X - x0) ** 2 / (x**2) + (Y - y0) ** 2 / (y**2))
    image = G * image

    if clip_to_input_range:
        image = torch.clamp(image, min=img_min, max=img_max)

    return image
