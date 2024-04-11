from yucca.image_processing.transforms.YuccaTransform import YuccaTransform
import numpy as np


class GibbsRinging(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        p_per_sample: float = 1.0,
        cut_freq=(96, 129),
        axes=(0, 3),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.cut_freq = cut_freq
        self.axes = axes
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(cut_freq, axes):
        cut_freq = np.random.randint(*cut_freq)
        axis = np.random.randint(*axes)
        return cut_freq, axis

    def __gibbsRinging__(self, image, num_sample, axis):
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
        if self.clip_to_input_range:
            image = np.clip(image, a_min=img_min, a_max=img_max)
        return image

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"

        for b in range(data_dict[self.data_key].shape[0]):
            for c in range(data_dict[self.data_key][b].shape[0]):
                if np.random.uniform() < self.p_per_sample:
                    cut_freq, axis = self.get_params(self.cut_freq, self.axes)
                    data_dict[self.data_key][b, c] = self.__gibbsRinging__(data_dict[self.data_key][b, c], cut_freq, axis)
        return data_dict
