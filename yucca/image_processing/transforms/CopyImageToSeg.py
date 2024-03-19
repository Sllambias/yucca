from yucca.image_processing.transforms.YuccaTransform import YuccaTransform


class CopyImageToSeg(YuccaTransform):
    def __init__(self, copy=False, data_key="image", label_key="label"):
        self.copy = copy
        self.data_key = data_key
        self.label_key = label_key

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __copy__(self, data_dict):
        data_dict[self.label_key] = data_dict[self.data_key].copy()
        return data_dict

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"
        if self.copy:
            data_dict = self.__copy__(data_dict)
        return data_dict
