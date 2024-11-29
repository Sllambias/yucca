import torch
import numpy as np
from typing import Optional
from yucca.pipeline.preprocessing.YuccaPreprocessor import YuccaPreprocessor


class ClassificationPreprocessor(YuccaPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set up for classification
        self.label_exists = True
        self.preprocess_label = False

    def reverse_preprocessing(self, images: torch.Tensor, image_properties: dict, num_classes: Optional[int] = None):
        """
        Expected shape of images are:
        (b, c, x)
        """
        image_properties["save_format"] = "txt"
        return images.float().cpu().numpy(), image_properties

    def cast_to_numpy_array(self, images: list, label=None, classification=False):
        canvas = np.empty(2, dtype="object")
        images = np.vstack([image[np.newaxis] for image in images])
        canvas[:] = [images, label]
        images = canvas
        return images
