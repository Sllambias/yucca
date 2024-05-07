import numpy as np


def bias_field(image, clip_to_input_range: bool = False):
    img_min = image.min()
    img_max = image.max()

    if len(image.shape) == 3:
        x, y, z = image.shape
        X, Y, Z = np.meshgrid(
            np.linspace(0, x, x, endpoint=False),
            np.linspace(0, y, y, endpoint=False),
            np.linspace(0, z, z, endpoint=False),
            indexing="ij",
        )
        x0 = np.random.randint(0, x)
        y0 = np.random.randint(0, y)
        z0 = np.random.randint(0, z)
        G = 1 - (np.power((X - x0), 2) / (x**2) + np.power((Y - y0), 2) / (y**2) + np.power((Z - z0), 2) / (z**2))
    else:
        x, y = image.shape
        X, Y = np.meshgrid(
            np.linspace(0, x, x, endpoint=False),
            np.linspace(0, y, y, endpoint=False),
            indexing="ij",
        )
        x0 = np.random.randint(0, x)
        y0 = np.random.randint(0, y)
        G = 1 - (np.power((X - x0), 2) / (x**2) + np.power((Y - y0), 2) / (y**2))
    image = np.multiply(G, image)
    if clip_to_input_range:
        image = np.clip(image, a_min=img_min, a_max=img_max)
    return image
