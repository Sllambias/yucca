import numpy as np


# Stolen from Batchgenerators to avoid import error caused by deprecated modules imported in
# Batchgenerators.
def augment_gamma(
    data_sample,
    gamma_range=(0.5, 2),
    invert_image=False,
    epsilon=1e-7,
    per_channel=False,
    clip_to_input_range=False,
):
    if invert_image:
        data_sample = -data_sample

    if not per_channel:
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        img_min = data_sample.min()
        img_max = data_sample.max()
        img_range = img_max - img_min
        data_sample = np.power(((data_sample - img_min) / float(img_range + epsilon)), gamma) * img_range + img_min
        if clip_to_input_range:
            data_sample = np.clip(data_sample, a_min=img_min, a_max=img_max)
    else:
        for c in range(data_sample.shape[0]):
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            img_min = data_sample[c].min()
            img_max = data_sample[c].max()
            img_range = img_max - img_min
            data_sample[c] = (
                np.power(((data_sample[c] - img_min) / float(img_range + epsilon)), gamma) * float(img_range + epsilon)
                + img_min
            )
            if clip_to_input_range:
                data_sample[c] = np.clip(data_sample[c], a_min=img_min, a_max=img_max)
    if invert_image:
        data_sample = -data_sample
    return data_sample
