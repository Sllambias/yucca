from yucca.image_processing.transforms.YuccaTransform import YuccaTransform
import numpy as np


class GibbsRinging(YuccaTransform):
    def __init__(self, data_key="image", p_per_sample=1, cutFreq=(96, 129), axes=(0, 3)):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.cutFreq = cutFreq
        self.axes = axes

    @staticmethod
    def get_params(cutFreq, axes):
        cutFreq = np.random.randint(*cutFreq)
        axis = np.random.randint(*axes)
        return cutFreq, axis

    def __gibbsRinging__(self, imageVolume, numSample, axis):
        m = min(0, imageVolume.min())
        imageVolume += abs(m)
        if len(imageVolume.shape) == 3:
            assert axis in [0, 1, 2], "Incorrect or no axis"

            h, w, d = imageVolume.shape
            if axis == 0:
                imageVolume = imageVolume.transpose(0, 2, 1)
                imageVolume = np.fft.fftshift(np.fft.fftn(imageVolume, s=[h, d, w]))
                imageVolume[:, :, 0 : int(np.ceil(w / 2) - np.ceil(numSample / 2))] = 0
                imageVolume[:, :, int(np.ceil(w / 2) + np.ceil(numSample / 2)) : w] = 0
                imageVolume = abs(np.fft.ifftn(np.fft.ifftshift(imageVolume), s=[h, d, w]))
                imageVolume = imageVolume.transpose(0, 2, 1)
            elif axis == 1:
                imageVolume = imageVolume.transpose(1, 2, 0)
                imageVolume = np.fft.fftshift(np.fft.fftn(imageVolume, s=[w, d, h]))
                imageVolume[:, :, 0 : int(np.ceil(h / 2) - np.ceil(numSample / 2))] = 0
                imageVolume[:, :, int(np.ceil(h / 2) + np.ceil(numSample / 2)) : h] = 0
                imageVolume = abs(np.fft.ifftn(np.fft.ifftshift(imageVolume), s=[w, d, h]))
                imageVolume = imageVolume.transpose(2, 0, 1)
            else:
                imageVolume = np.fft.fftshift(np.fft.fftn(imageVolume, s=[h, w, d]))
                imageVolume[:, :, 0 : int(np.ceil(d / 2) - np.ceil(numSample / 2))] = 0
                imageVolume[:, :, int(np.ceil(d / 2) + np.ceil(numSample / 2)) : d] = 0
                imageVolume = abs(np.fft.ifftn(np.fft.ifftshift(imageVolume), s=[h, w, d]))
        elif len(imageVolume.shape) == 2:
            assert axis in [0, 1], "incorrect or no axis"
            h, w = imageVolume.shape
            if axis == 0:
                imageVolume = np.fft.fftshift(np.fft.fftn(imageVolume, s=[h, w]))
                imageVolume[:, 0 : int(np.ceil(w / 2) - np.ceil(numSample / 2))] = 0
                imageVolume[:, int(np.ceil(w / 2) + np.ceil(numSample / 2)) : w] = 0
                imageVolume = abs(np.fft.ifftn(np.fft.ifftshift(imageVolume), s=[h, w]))
            else:
                imageVolume = imageVolume.conj().T
                imageVolume = np.fft.fftshift(np.fft.fftn(imageVolume, s=[w, h]))
                imageVolume[:, 0 : int(np.ceil(h / 2) - np.ceil(numSample / 2))] = 0
                imageVolume[:, int(np.ceil(h / 2) + np.ceil(numSample / 2)) : h] = 0
                imageVolume = abs(np.fft.ifftn(np.fft.ifftshift(imageVolume), s=[w, h]))
                imageVolume = imageVolume.conj().T
        imageVolume -= m
        return imageVolume

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"

        for b in range(data_dict[self.data_key].shape[0]):
            for c in range(data_dict[self.data_key][b].shape[0]):
                if np.random.uniform() < self.p_per_sample:
                    cutFreq, axis = self.get_params(self.cutFreq, self.axes)
                    data_dict[self.data_key][b, c] = self.__gibbsRinging__(data_dict[self.data_key][b, c], cutFreq, axis)
        return data_dict
