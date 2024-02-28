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
    def preprocess_train_subject(self, subject_id):
        subject_id = subject_id.split(os.extsep, 1)[0]
        arraypath = join(self.target_dir, subject_id + ".npy")
        picklepath = join(self.target_dir, subject_id + ".pkl")

        if isfile(arraypath) and isfile(picklepath):
            logging.info(f"Case: {subject_id} already exists. Skipping.")
            return

        start_time = time.time()

        images, label, image_props = self._preprocess_train_subject(subject_id, label_exists=True, preprocess_label=False)

        images = np.array((np.array(images).T, label), dtype="object")
        images[0] = images[0].T

        # save the image
        np.save(arraypath, images)

        # save metadata as .pkl
        save_pickle(image_props, picklepath)

        end_time = time.time()
        logging.info(
            f"Preprocessed case: {subject_id} \n"
            f"size before: {image_props['original_size']} size after: {image_props['new_size']} \n"
            f"spacing before: {image_props['original_spacing']} spacing after: {image_props['new_spacing']} \n"
            f"Saving {subject_id} in {arraypath} \n"
            f"Time elapsed: {round(end_time-start_time, 4)} \n"
        )

    def reverse_preprocessing(self, images: torch.Tensor, image_properties: dict):
        """
        Expected shape of images are:
        (b, c, x)
        """
        image_properties["save_format"] = "txt"
        return images.cpu().numpy(), image_properties
