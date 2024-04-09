from yucca.image_processing.transforms.YuccaTransform import YuccaTransform
import numpy as np


class BiasField(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        p_per_sample=1,
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __biasField__(self, image):
        img_min = image.min()
        img_max = image.max()

        if len(image.shape) == 3:
            x, y, z = image.shape
            X, Y, Z = np.meshgrid(
                np.linspace(0, x, x, endpoint=False),
                np.linspace(0, y, y, endpoint=False),
                np.linspace(0, z, z, endpoint=False),
                indexing="ij",
            )
            x0 = np.random.randint(0, x)
            y0 = np.random.randint(0, y)
            z0 = np.random.randint(0, z)
            G = 1 - (np.power((X - x0), 2) / (x**2) + np.power((Y - y0), 2) / (y**2) + np.power((Z - z0), 2) / (z**2))
        else:
            x, y = image.shape
            X, Y = np.meshgrid(
                np.linspace(0, x, x, endpoint=False),
                np.linspace(0, y, y, endpoint=False),
                indexing="ij",
            )
            x0 = np.random.randint(0, x)
            y0 = np.random.randint(0, y)
            G = 1 - (np.power((X - x0), 2) / (x**2) + np.power((Y - y0), 2) / (y**2))
        image = np.multiply(G, image)
        if self.clip_to_input_range:
            image = np.clip(image, a_min=img_min, a_max=img_max)
        return image

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape. \nShould be (b, c, x, y, z) or (b, c, x, y) and is:\
                {data_dict[self.data_key].shape}"

        for b in range(data_dict[self.data_key].shape[0]):
            for c in range(data_dict[self.data_key][b].shape[0]):
                if np.random.uniform() < self.p_per_sample:
                    data_dict[self.data_key][b, c] = self.__biasField__(data_dict[self.data_key][b, c])
        return data_dict
