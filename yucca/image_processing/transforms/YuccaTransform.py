from batchgenerators.transforms.abstract_transforms import AbstractTransform
from abc import abstractmethod


class YuccaTransform(AbstractTransform):
    @abstractmethod
    def get_params(self):
        """
        This will return a random value between
        values in a Tuple(low, high)
        e.g.
            sigma = (0, 10)
            val = np.random.uniform(*sigma)

        This is done for each random
        parameter of the augmentation.

        This method can be called
        (1) once for all samples of a batch
        (2) individually for each sample
        (3) individually for each modality of a sample

        This allows a high degree of flexibility.
        In label-preserving transforms we often use (1)
        to create a higher degree of variation
        In transforms that are not label-preserving we
        use either (2) or (3) with (2) being the
        preferred choice.
        This allows different intensities for each
        sample in a batch, while using the same
        intensity for all modalities of a sample
        and for both image and label.
        This is extremely important for transforms
        such as rotation, where we must ensure
        data modalities and labels remain registered.
        """

    @abstractmethod
    def __call__(self):
        """
        This will be of the form __call__(self, packed_dict: dict = None, **unpacked_dict):
        which allows calling it as either transform(data_dict) or transform(**data_dict),
        supporting both Torch pipelines and batchgenerators.
        """
