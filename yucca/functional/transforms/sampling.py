import numpy as np
from skimage.transform import resize


def downsample_label(label, factors):
    orig_type = label.dtype
    orig_shape = label.shape
    downsampled_labels = []
    for factor in factors:
        target_shape = np.array(orig_shape).astype(int)
        for i in range(2, len(orig_shape)):
            target_shape[i] *= factor
        if np.all(target_shape == orig_shape):
            downsampled_labels.append(label)
        else:
            canvas = np.zeros(target_shape)
            for b in range(label.shape[0]):
                for c in range(label[b].shape[0]):
                    canvas[b, c] = resize(
                        label[b, c].astype(float),
                        target_shape[2:],
                        0,
                        mode="edge",
                        clip=True,
                        anti_aliasing=False,
                    ).astype(orig_type)
            downsampled_labels.append(canvas)
    return downsampled_labels


def simulate_lowres(image, target_shape, clip_to_input_range):
    img_min = image.min()
    img_max = image.max()
    shape = image.shape
    downsampled = resize(
        image.astype(float),
        target_shape,
        order=0,
        mode="edge",
        anti_aliasing=False,
    )
    image = resize(downsampled, shape, order=3, mode="edge", anti_aliasing=False)
    if clip_to_input_range:
        image = np.clip(image, a_min=img_min, a_max=img_max)
    return image
