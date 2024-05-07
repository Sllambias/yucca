import numpy as np


def motion_ghosting(image, alpha, num_reps, axis, clip_to_input_range):
    img_min = image.min()
    img_max = image.max()
    m = min(0, img_min)
    image += abs(m)
    if len(image.shape) == 3:
        assert axis in [0, 1, 2], "Incorrect or no axis"

        h, w, d = image.shape

        image = np.fft.fftn(image, s=[h, w, d])

        if axis == 0:
            image[0:-1:num_reps, :, :] = alpha * image[0:-1:num_reps, :, :]
        elif axis == 1:
            image[:, 0:-1:num_reps, :] = alpha * image[:, 0:-1:num_reps, :]
        else:
            image[:, :, 0:-1:num_reps] = alpha * image[:, :, 0:-1:num_reps]

        image = abs(np.fft.ifftn(image, s=[h, w, d]))
    if len(image.shape) == 2:
        assert axis in [0, 1], "Incorrect or no axis"
        h, w = image.shape
        image = np.fft.fftn(image, s=[h, w])

        if axis == 0:
            image[0:-1:num_reps, :] = alpha * image[0:-1:num_reps, :]
        else:
            image[:, 0:-1:num_reps] = alpha * image[:, 0:-1:num_reps]
        image = abs(np.fft.ifftn(image, s=[h, w]))
    image -= abs(m)
    if clip_to_input_range:
        image = np.clip(image, a_min=img_min, a_max=img_max)
    return image
