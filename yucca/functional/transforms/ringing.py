import numpy as np


def gibbs_ringing(image, num_sample, axis, clip_to_input_range: bool = False):
    img_min = image.min()
    img_max = image.max()
    m = min(0, img_min)
    image += abs(m)
    if len(image.shape) == 3:
        assert axis in [0, 1, 2], "Incorrect or no axis"

        h, w, d = image.shape
        if axis == 0:
            image = image.transpose(0, 2, 1)
            image = np.fft.fftshift(np.fft.fftn(image, s=[h, d, w]))
            image[:, :, 0 : int(np.ceil(w / 2) - np.ceil(num_sample / 2))] = 0
            image[:, :, int(np.ceil(w / 2) + np.ceil(num_sample / 2)) : w] = 0
            image = abs(np.fft.ifftn(np.fft.ifftshift(image), s=[h, d, w]))
            image = image.transpose(0, 2, 1)
        elif axis == 1:
            image = image.transpose(1, 2, 0)
            image = np.fft.fftshift(np.fft.fftn(image, s=[w, d, h]))
            image[:, :, 0 : int(np.ceil(h / 2) - np.ceil(num_sample / 2))] = 0
            image[:, :, int(np.ceil(h / 2) + np.ceil(num_sample / 2)) : h] = 0
            image = abs(np.fft.ifftn(np.fft.ifftshift(image), s=[w, d, h]))
            image = image.transpose(2, 0, 1)
        else:
            image = np.fft.fftshift(np.fft.fftn(image, s=[h, w, d]))
            image[:, :, 0 : int(np.ceil(d / 2) - np.ceil(num_sample / 2))] = 0
            image[:, :, int(np.ceil(d / 2) + np.ceil(num_sample / 2)) : d] = 0
            image = abs(np.fft.ifftn(np.fft.ifftshift(image), s=[h, w, d]))
    elif len(image.shape) == 2:
        assert axis in [0, 1], "incorrect or no axis"
        h, w = image.shape
        if axis == 0:
            image = np.fft.fftshift(np.fft.fftn(image, s=[h, w]))
            image[:, 0 : int(np.ceil(w / 2) - np.ceil(num_sample / 2))] = 0
            image[:, int(np.ceil(w / 2) + np.ceil(num_sample / 2)) : w] = 0
            image = abs(np.fft.ifftn(np.fft.ifftshift(image), s=[h, w]))
        else:
            image = image.conj().T
            image = np.fft.fftshift(np.fft.fftn(image, s=[w, h]))
            image[:, 0 : int(np.ceil(h / 2) - np.ceil(num_sample / 2))] = 0
            image[:, int(np.ceil(h / 2) + np.ceil(num_sample / 2)) : h] = 0
            image = abs(np.fft.ifftn(np.fft.ifftshift(image), s=[w, h]))
            image = image.conj().T
    image -= abs(m)
    if clip_to_input_range:
        image = np.clip(image, a_min=img_min, a_max=img_max)
    return image
