import numpy as np


def remove_small_objects(array: np.ndarray, min_size: float | int, voxel_spacing: np.ndarray | list):
    """
    This function will remove any connected components smaller
    than the minimum size in mm
    """
    labels, counts = np.unique(array, return_counts=True)
    labels = list(labels)

    # Remove background
    labels.remove(0)

    voxel_size = np.prod(voxel_spacing)

    for label in labels:
        if not voxel_size * counts[label] >= min_size:
            array[array == label] = 0
            labels.remove(label)

    return array, labels
