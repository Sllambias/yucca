from yucca.modules.data.datasets.YuccaDataset import YuccaTrainDataset
import numpy as np


class YuccaTrainDataset_1modality(YuccaTrainDataset):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

    def unpack(self, data, supervised: bool):
        images, label = super().unpack(data, supervised)
        chosen_modality = np.random.randint(0, images.shape[0])

        # by slicing this way we keep the channel dimension
        # rather than doing images[chosen_modality] which makes (c,h,w,d) --> (h,w,d)
        images = images[chosen_modality : chosen_modality + 1]
        return images, label

    def unpack_with_zeros(self, data, supervised: bool):
        images, label = super().unpack_with_zeros(data, supervised)

        # first find a modality that is not a zero-array
        valid_modalities = [i for i in range(images.shape[0]) if np.any(data[i])]

        chosen_modality = np.random.choice(valid_modalities)
        images = images[chosen_modality : chosen_modality + 1]
        return images, label
