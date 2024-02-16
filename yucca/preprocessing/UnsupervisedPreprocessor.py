"""
Takes raw data conforming with Yucca standards and preprocesses according to the generic scheme
"""

import numpy as np
import re
import os
import logging
import time
from yucca.preprocessing.YuccaPreprocessor import YuccaPreprocessor
from yucca.paths import yucca_preprocessed_data, yucca_raw_data
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    subfiles,
    save_pickle,
    isfile,
)


class UnsupervisedPreprocessor(YuccaPreprocessor):
    def initialize_paths(self):
        # Have to overwrite how we get the subject_ids as there's no labelsTr to get them from.
        # Therefore we use the imagesTr folder and remove the modality suffix.
        self.target_dir = join(yucca_preprocessed_data, self.task, self.plans["plans_name"])
        self.input_dir = join(yucca_raw_data, self.task)
        self.imagepaths = subfiles(join(self.input_dir, "imagesTr"), suffix=self.image_extension)

        subject_ids = subfiles(join(self.input_dir, "imagesTr"), suffix=self.image_extension, join=False)
        self.subject_ids = [re.sub(r"_\d+\.", ".", subject) for subject in subject_ids]

    def preprocess_train_subject(self, subject_id):
        subject_id = subject_id.split(os.extsep, 1)[0]
        arraypath = join(self.target_dir, subject_id + ".npy")
        picklepath = join(self.target_dir, subject_id + ".pkl")

        if isfile(arraypath) and isfile(picklepath):
            logging.info(f"Case: {subject_id} already exists. Skipping.")
            return

        start_time = time.time()

        images, _, image_props = self._preprocess_train_subject(subject_id, label_exists=False, preprocess_label=False)

        images = np.array(images)

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
