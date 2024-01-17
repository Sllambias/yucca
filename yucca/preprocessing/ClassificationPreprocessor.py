"""
Takes raw data conforming with Yucca standards and preprocesses according to the generic scheme
"""
import numpy as np
import torch
import nibabel as nib
import os
import logging
from yucca.preprocessing.YuccaPreprocessor import YuccaPreprocessor
from yucca.utils.nib_utils import get_nib_spacing, get_nib_orientation, reorient_nib_image
from yucca.utils.type_conversions import nifti_or_np_to_np
from yucca.utils.loading import read_file_to_nifti_or_np
from yucca.image_processing.objects.BoundingBox import get_bbox_for_foreground
from yucca.image_processing.cropping_and_padding import crop_to_box, pad_to_size
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    subfiles,
    save_pickle,
    isfile,
)


class ClassificationPreprocessor(YuccaPreprocessor):
    def preprocess_train_subject(self, subject_id):
        images, label, image_props = self._preprocess_train_subject(subject_id, label_exists=True, preprocess_label=False)

        images = np.array((np.array(images).T, label), dtype="object")
        images[0] = images[0].T

        logging.info(
            f"size before: {image_props['original_size']} size after: {image_props['new_size']} \n"
            f"spacing before: {image_props['original_spacing']} spacing after: {image_props['new_spacing']} \n"
            f"Saving {subject_id} in {image_props['arraypath']} \n"
        )

        # save the image
        np.save(image_props["arraypath"], images)

        # save metadata as .pkl
        save_pickle(image_props, image_props["picklepath"])

    def reverse_preprocessing(self, images: torch.Tensor, image_properties: dict):
        """
        Expected shape of images are:
        (b, c, x)
        """
        image_properties["save_format"] = "txt"
        return images.cpu().numpy(), image_properties
