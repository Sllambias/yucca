"""
Takes raw data conforming with Yucca standards and preprocesses according to the generic scheme
"""

import numpy as np
import torch
import os
import logging
import time
from yucca.preprocessing.YuccaPreprocessor import YuccaPreprocessor
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    save_pickle,
    isfile,
)


class ClassificationPreprocessor(YuccaPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set up for classification
        self.classification = True
        self.label_exists = True
        self.preprocess_label = False

    def reverse_preprocessing(self, images: torch.Tensor, image_properties: dict):
        """
        Expected shape of images are:
        (b, c, x)
        """
        image_properties["save_format"] = "txt"
        return images.cpu().numpy(), image_properties
