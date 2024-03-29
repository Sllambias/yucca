from yucca.image_processing.transforms.YuccaTransform import YuccaTransform
from yucca.preprocessing.normalization import normalizer


class Normalize(YuccaTransform):
    def __init__(self, normalize: bool = False, data_key: str = "image", scheme: str = "volume_wise_znorm"):
        assert scheme in [
            "255to1",
            "clip",
            "volume_wise_znorm",
        ], f"Scheme {scheme} not supported, only schemes which are not relying on dataset level properties are currently supported"

        self.normalize = normalize
        self.data_key = data_key
        self.scheme = scheme

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __normalize__(self, data_dict):
        data_dict[self.data_key] = normalizer(data_dict[self.data_key], scheme=self.scheme)
        return data_dict

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"

        if self.normalize:
            data_dict = self.__normalize__(data_dict)
        return data_dict
