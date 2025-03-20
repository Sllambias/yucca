import torch
import os
import numpy as np
import re
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


class ClassificationPreprocessorWithCovariates(ClassificationPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_paths(self):
        super().initialize_paths()
        self.covariatepathsTr = os.path.join(self.input_dir, "covariatesTr")
        self.covariatepathsTs = os.path.join(self.input_dir, "covariatesTs")

    def _preprocess_train_subject(self, subject_id, label_exists: bool, preprocess_label: bool):
        images, label, image_props = super()._preprocess_train_subject(subject_id, label_exists, preprocess_label)
        covariates = np.loadtxt(os.path.join(self.covariatepathsTr, re.escape(subject_id) + "_COV.txt"))
        label = np.array([covariates, label], dtype="object")
        return images, label, image_props

    def cast_to_numpy_array(self, images: list, label=None, classification=False):
        # In this scenario the labels will also contain the covariates

        canvas = np.empty(3, dtype="object")
        images = np.vstack([image[np.newaxis] for image in images])
        canvas[:] = [images, label[0], label[-1]]
        images = canvas
        return images
