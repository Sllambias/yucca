from copy import deepcopy
import numpy as np


def get_bbox_for_foreground(array, background_label=0):
    array = deepcopy(array)
    array[array > background_label] = 1
    return get_bbox_for_label(array, label=1)


def get_bbox_for_label(array, label, padding=0):
    a = np.where(array == label)

    x1, x2, y1, y2 = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])

    xmin = min(x1, x2) - padding
    xmax = max(x1, x2) + padding

    ymin = min(y1, y2) - padding
    ymax = max(y1, y2) + padding

    if len(array.shape) == 3:
        z1, z2 = np.min(a[2]), np.max(a[2])

        zmin = min(z1, z2) - padding
        zmax = max(z1, z2) + padding

        return [xmin, xmax + 1, ymin, ymax + 1, zmin, zmax + 1]

    return [xmin, xmax + 1, ymin, ymax + 1]
