import numpy as np
from scipy.ndimage import gaussian_filter
import math as m


def create_zero_centered_coordinate_matrix(shape):
    if len(shape) == 3:
        mesh = np.array(
            np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing="ij",
            )
        ).astype(float)
    if len(shape) == 2:
        mesh = np.array(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")).astype(float)

    for d in range(len(shape)):
        mesh[d] -= (mesh.shape[d + 1] - 1) / 2
        assert np.mean(mesh[d]) == 0, "beware: mesh didnt zero-center"
    return mesh


def deform_coordinate_matrix(coordinate_matrix, alpha, sigma):
    deforms = np.array(
        [
            gaussian_filter(
                (np.random.random(coordinate_matrix.shape[1:]) * 2 - 1),
                sigma,
                mode="constant",
                cval=0,
            )
            * alpha
            for _ in range(coordinate_matrix.shape[0])
        ]
    )
    coordinate_matrix = deforms + coordinate_matrix
    return coordinate_matrix


def Rx(theta):
    return np.array([[1, 0, 0], [0, m.cos(theta), -m.sin(theta)], [0, m.sin(theta), m.cos(theta)]])


def Ry(theta):
    return np.array([[m.cos(theta), 0, m.sin(theta)], [0, 1, 0], [-m.sin(theta), 0, m.cos(theta)]])


def Rz(theta):
    return np.array([[m.cos(theta), -m.sin(theta), 0], [m.sin(theta), m.cos(theta), 0], [0, 0, 1]])


def Rz2D(theta):
    return np.array([[m.cos(theta), -m.sin(theta)], [m.sin(theta), m.cos(theta)]])


def get_max_rotated_size(patch_size):
    if len(patch_size) == 2:
        max_dim = int(np.sqrt(patch_size[0] ** 2 + patch_size[1] ** 2))
        return (max_dim, max_dim)

    max_dim_0 = max(
        int(np.sqrt(patch_size[0] ** 2 + patch_size[1] ** 2)),
        int(np.sqrt(patch_size[0] ** 2 + patch_size[2] ** 2)),
    )

    max_dim_1 = max(
        int(np.sqrt(patch_size[1] ** 2 + patch_size[0] ** 2)),
        int(np.sqrt(patch_size[1] ** 2 + patch_size[2] ** 2)),
    )

    max_dim_2 = max(
        int(np.sqrt(patch_size[2] ** 2 + patch_size[0] ** 2)),
        int(np.sqrt(patch_size[2] ** 2 + patch_size[1] ** 2)),
    )

    return (max_dim_0, max_dim_1, max_dim_2)
