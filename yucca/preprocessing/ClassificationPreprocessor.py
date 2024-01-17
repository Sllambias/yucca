"""
Takes raw data conforming with Yucca standards and preprocesses according to the generic scheme
"""
import numpy as np
import torch
import nibabel as nib
import os
import logging
from yucca.preprocessing.YuccaPreprocessor import YuccaPreprocessor
from yucca.paths import yucca_preprocessed_data, yucca_raw_data
from yucca.preprocessing.normalization import normalizer
from yucca.utils.nib_utils import get_nib_spacing, get_nib_orientation, reorient_nib_image
from yucca.utils.type_conversions import nifti_or_np_to_np
from yucca.utils.loading import read_file_to_nifti_or_np
from yucca.image_processing.objects.BoundingBox import get_bbox_for_foreground
from yucca.image_processing.cropping_and_padding import crop_to_box, pad_to_size
from multiprocessing import Pool
from skimage.transform import resize
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    subfiles,
    save_pickle,
    maybe_mkdir_p,
    isfile,
    subdirs,
)


class ClassificationPreprocessor(YuccaPreprocessor):
    def _preprocess_train_subject(self, subject_id):
        image_props = {}
        subject_id = subject_id.split(os.extsep, 1)[0]
        print(f"Preprocessing: {subject_id}")
        arraypath = join(self.target_dir, subject_id + ".npy")
        picklepath = join(self.target_dir, subject_id + ".pkl")

        if isfile(arraypath) and isfile(picklepath):
            print(f"Case: {subject_id} already exists. Skipping.")
            return
        # First find relevant images by their paths and save them in the image property pickle
        # Then load them as images
        # The '_' in the end is to avoid treating Case_4_000 AND Case_42_000 as different versions
        # of the label named Case_4 as both would start with "Case_4", however only the correct one is
        # followed by an underscore
        imagepaths = [impath for impath in self.imagepaths if os.path.split(impath)[-1].startswith(subject_id + "_")]

        image_props["image files"] = imagepaths
        images = [read_file_to_nifti_or_np(image) for image in imagepaths]

        # Do the same with label
        label = [
            labelpath
            for labelpath in subfiles(join(self.input_dir, "labelsTr"))
            if os.path.split(labelpath)[-1].startswith(subject_id + ".")
        ]
        assert len(label) < 2, f"unexpected number of labels found. Expected 1 or 0 and found {len(label)}"
        image_props["label file"] = label[0]
        label = read_file_to_nifti_or_np(label[0], dtype=np.uint8)

        if not self.disable_sanity_checks:
            self.run_sanity_checks(images, label, subject_id, imagepaths)

        original_size = np.array(images[0].shape)
        (
            images,
            original_spacing,
            original_orientation,
            final_direction,
            label,
        ) = self.apply_nifti_preprocessing_and_return_numpy(images, original_size, label)

        # Cropping is performed to save computational resources. We are only removing background.
        if self.plans["crop_to_nonzero"]:
            nonzero_box = get_bbox_for_foreground(images[0], background_label=0)
            image_props["crop_to_nonzero"] = nonzero_box
            for i in range(len(images)):
                images[i] = crop_to_box(images[i], nonzero_box)
        else:
            image_props["crop_to_nonzero"] = self.plans["crop_to_nonzero"]

        images = self.transpose_case(images, self.transpose_forward, label=None)

        resample_target_size, final_target_size = self.determine_target_size(
            images_transposed=images,
            original_spacing=original_spacing,
            transpose_forward=self.transpose_forward,
        )

        images = self._resample_and_normalize_case(
            images=images,
            target_size=resample_target_size,
            label=None,
            norm_op=self.plans["normalization_scheme"],
        )

        if final_target_size is not None:
            images = self.pad_to_size(images, size=final_target_size, label=None)

        images = np.array((np.array(images).T, label), dtype="object")
        images[0] = images[0].T
        final_size = list(images[0][0].shape)

        # For classification there's no foreground classes
        # And no connected components to analyze.
        foreground_locs = []
        label_cc_n = label_cc_sizes = 0

        # save relevant values
        image_props["original_spacing"] = original_spacing
        image_props["original_size"] = original_size
        image_props["original_orientation"] = original_orientation
        image_props["new_spacing"] = self.target_spacing
        image_props["new_size"] = final_size
        image_props["new_direction"] = final_direction
        image_props["foreground_locations"] = foreground_locs
        image_props["label_cc_n"] = label_cc_n
        image_props["label_cc_sizes"] = label_cc_sizes

        logging.info(
            f"size before: {original_size} size after: {image_props['new_size']} \n"
            f"spacing before: {original_spacing} spacing after: {image_props['new_spacing']} \n"
            f"Saving {subject_id} in {arraypath} \n"
        )

        # save the image
        np.save(arraypath, images)

        # save metadata as .pkl
        save_pickle(image_props, picklepath)

    def reverse_preprocessing(self, images: torch.Tensor, image_properties: dict):
        """
        Expected shape of images are:
        (b, c, x)
        """
        image_properties["save_format"] = "txt"
        return images.cpu().numpy(), image_properties
