from yucca.modules.data.augmentation.transforms.YuccaTransform import YuccaTransform
from yucca.functional.array_operations.normalization import normalizer


class Normalize(YuccaTransform):
    def __init__(
        self, normalize: bool = False, data_key: str = "image", metadata_key="metadata", scheme: str = "volume_wise_znorm"
    ):
        assert scheme in [
            "255to1",
            "minmax",
            "range",
            "clip",
            "volume_wise_znorm",
        ], f"Scheme {scheme} not supported, only schemes which are not relying on dataset level properties are currently supported"

        self.normalize = normalize
        self.data_key = data_key
        self.metadata_key = metadata_key
        self.scheme = scheme

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __normalize__(self, data_dict):
        for b in range(data_dict[self.data_key].shape[0]):
            for c in range(data_dict[self.data_key].shape[1]):
                data_dict[self.data_key][b, c] = normalizer(
                    data_dict[self.data_key][b, c], scheme=self.scheme, intensities=data_dict[self.metadata_key]
                )
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
