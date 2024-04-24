import numpy as np


def transpose_case(images: list, axes):
    for i in range(len(images)):
        assert len(images[i].shape) == len(axes), (
            "image and transpose axes do not match. \n"
            f"images[i].shape == {images[i].shape} \n"
            f"transpose == {axes} \n"
            f"len(images[i].shape) == {len(images[i]).shape} \n"
            f"len(transpose) == {len(axes)} \n"
        )
        images[i] = images[i].transpose(axes)
    return images


def transpose_array(image: np.ndarray, axes):
    return image.transpose(axes)
