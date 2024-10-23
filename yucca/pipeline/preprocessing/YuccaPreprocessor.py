import numpy as np
import torch
import nibabel as nib
import os
import logging
import time
import re
from typing import Optional
from yucca.functional.testing.data.nifti import (
    verify_spacing_is_equal,
    verify_orientation_is_equal,
)
from yucca.functional.testing.data.array import verify_labels_are_equal, verify_array_shape_is_equal
from yucca.functional.preprocessing import (
    preprocess_case_for_inference,
    preprocess_case_for_training_with_label,
    preprocess_case_for_training_without_label,
    reverse_preprocessing,
)
from yucca.functional.utils.loading import load_yaml, read_file_to_nifti_or_np
from yucca.paths import get_preprocessed_data_path, get_raw_data_path
from multiprocessing import Pool
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    subfiles,
    save_pickle,
    maybe_mkdir_p as ensure_dir_exists,
    isfile,
)


class YuccaPreprocessor(object):
    """
    The YuccaPreprocessor class is designed to preprocess medical images for the Yucca project.
    It implements various preprocessing steps, such as reorientation, cropping, normalization, and resizing,
    based on the plans specified in an YuccaPlanner.

    For training the _preprocess_train_subject method prepares input images for the Yucca model.
    The preprocess_case_for_inference method prepares input images for the Yucca model during the inference phase,
    ensuring that they match the requirements specified during training.
    The reverse_preprocessing method is then used to revert the processed images back to their original form,
    allowing for a meaningful interpretation of the model's predictions.
    These methods collectively provide a consistent and reversible preprocessing pipeline for both training and inference.

    The operations that can be enabled/defined in the YuccaPlanner and carried out by the
    YuccaPreprocessor are:

    (1) The starting orientation - defaults to RAS (for medical images).
    (2) The cropping operation - defaults to crop to nonzero bounding box
    (3) The Transposition operation (along with the reverse transpose operation,
    to be used during inference) - defaults to no transposition if image dimensions and spacings
    are not too anisotropic.
    (4) The Resample operation - defaults to resampling to the median spacing of the dataset.
    (5) The Normalization operation - defaults to standardization = (image - mean) / std
    per modality to preserve ranges to account for CT pixel values representing specific physical
    attributes.

    Additionally it carries out a number of tests and analyzes each image for foreground locations
    which is used later to oversample foreground.
    """

    def __init__(
        self,
        plans_path,
        task=None,
        threads=None,
        disable_sanity_checks=False,
        enable_cc_analysis=False,
        allow_missing_modalities=False,
        compress=False,
        get_foreground_locs_per_label=False,
        preprocess_test=False,
        sliding_window_prediction=True,
    ):
        self.name = str(self.__class__.__name__)
        self.task = task
        self.plans_path = plans_path
        self.plans = self.load_plans(plans_path)
        self.threads = threads
        self.disable_sanity_checks = disable_sanity_checks
        self.enable_cc_analysis = enable_cc_analysis
        self.allow_missing_modalities = allow_missing_modalities
        self.compress = compress
        self.get_foreground_locs_per_label = get_foreground_locs_per_label
        self.preprocess_test = preprocess_test
        self.sliding_window_prediction = sliding_window_prediction

        # lists for information we would like to attain
        self.transpose_forward = []
        self.transpose_backward = []
        self.target_spacing = []

        # set up for segmentation
        self.classification = False
        self.label_exists = True
        self.preprocess_label = True

    def initialize_paths(self):
        self.target_dir = join(get_preprocessed_data_path(), self.task, self.plans["plans_name"])
        self.input_dir = join(get_raw_data_path(), self.task)
        self.imagepaths = subfiles(join(self.input_dir, "imagesTr"), suffix=self.image_extension)
        self.subject_ids = [
            file for file in subfiles(join(self.input_dir, "labelsTr"), join=False) if not file.startswith(".")
        ]
        if self.preprocess_test:
            self.test_imagepaths = subfiles(join(self.input_dir, "imagesTs"), suffix=self.image_extension)
            self.test_subject_ids = [
                file for file in subfiles(join(self.input_dir, "labelsTs"), join=False) if not file.startswith(".")
            ]
            self.test_target_dir = join(get_preprocessed_data_path(), self.task + "_TEST", self.plans["plans_name"])

    def initialize_properties(self):
        """
        here we basically set up things that are needed for preprocessing during training,
        but that aren't necessary during inference
        """
        self.dataset_properties = self.plans["dataset_properties"]
        self.intensities = self.dataset_properties["intensities"]
        self.image_extension = self.dataset_properties.get("image_extension") or "nii.gz"

        if self.dataset_properties.get("background_pixel_values") is not None:
            self.background_value_for_first_modality = self.dataset_properties.get("background_pixel_values")[0]
        else:
            self.background_value_for_first_modality = 0

        # op values
        self.transpose_forward = np.array(self.plans["transpose_forward"], dtype=int)
        self.transpose_backward = np.array(self.plans["transpose_backward"], dtype=int)
        self.target_spacing = self.plans["target_spacing"]
        self.target_size = (
            np.array(self.plans.get("target_size"), dtype=int) if self.plans.get("target_size") not in ["null", None] else None
        )

    def run(self):
        self.initialize_properties()
        self.initialize_paths()
        ensure_dir_exists(self.target_dir)
        self.verify_compression_level(self.target_dir, self.compress)

        logging.info(
            f"{'Preprocessing Task:':25.25} {self.task} \n"
            f"{'Using Planner:':25.25} {self.plans_path} \n"
            f"{'Crop to nonzero:':25.25} {self.plans['crop_to_nonzero']} \n"
            f"{'Normalization scheme:':25.25} {self.plans['normalization_scheme']} \n"
            f"{'Transpose Forward:':25.25} {self.transpose_forward} \n"
            f"{'Transpose Backward:':25.25} {self.transpose_backward} \n"
            f"{'Number of threads:':25.25} {self.threads}"
        )
        p = Pool(self.threads)
        p.map(self.preprocess_train_subject, self.subject_ids)
        p.close()
        p.join()

        if self.preprocess_test:
            ensure_dir_exists(self.test_target_dir)
            p = Pool(self.threads)
            p.map(self.preprocess_test_subject, self.test_subject_ids)
            p.close()
            p.join()

    def preprocess_train_subject(self, subject_id):
        """
        This is the bread and butter of the preprocessor.
        The following steps are taken:

        (1) Load Images:
        Extract relevant image files associated with the given subject_id.
        Load the images using the nibabel library.

        (2) Reorientation (Optional):
        Check if valid qform or sform codes are present in the header.
        If valid, reorient the images to the target orientation specified in the plans.
        Update the original and new orientation information in the image_props dictionary.

        (3) Normalization and Transposition:
        Normalize each image based on the specified normalization scheme and intensities.
        Transpose the images according to the forward transpose axes specified in the plans.

        (4) Cropping (Optional):
        If the crop_to_nonzero option is enabled in the plans, crop the images to the nonzero bounding box.
        Update the image_props dictionary with cropping information.

        (5) Resampling:
        Resample images to the target spacing specified in the plans.
        Update the image_props dictionary with original and new spacing information.

        (6) Foreground Locations:
        Extract some locations of the foreground, which will be used in oversampling of foreground classes.
        Determine the number and sizes of connected components in the ground truth label (can be used in analysis).

        (7) Save Preprocessed Data:
        Stack the preprocessed images and label.
        Save the preprocessed data as a NumPy array in a .npy file.
        Save relevant metadata as a .pkl file.

        (8) Print Information:
        Print information about the size and spacing before and after preprocessing.
        Print the path where the preprocessed data is saved.
        """
        subject_id = subject_id.split(os.extsep, 1)[0]
        if self.compress:
            arraypath = join(self.target_dir, subject_id + ".npz")
        else:
            arraypath = join(self.target_dir, subject_id + ".npy")
        picklepath = join(self.target_dir, subject_id + ".pkl")
        if isfile(arraypath) and isfile(picklepath):
            logging.info(f"Case: {subject_id} already exists. Skipping.")
            return

        start_time = time.time()

        images, label, image_props = self._preprocess_train_subject(
            subject_id, label_exists=self.label_exists, preprocess_label=self.preprocess_label
        )
        images = self.cast_to_numpy_array(images=images, label=label, classification=self.classification)

        # save the image
        if self.compress:
            np.savez_compressed(arraypath, data=images)
        else:
            np.save(arraypath, images)

        # save metadata as .pkl
        save_pickle(image_props, picklepath)

        end_time = time.time()
        logging.info(
            f"Preprocessed case: {subject_id} \n"
            f"size before: {image_props['original_size']} \n"
            f"cropping enabled: {self.plans['crop_to_nonzero']} \n"
            f"size after crop: {image_props['size_before_transpose']} \n"
            f"size final: {image_props['new_size']} \n"
            f"spacing before: {image_props['original_spacing']} spacing after: {image_props['new_spacing']} \n"
            f"Saving {subject_id} in {arraypath} \n"
            f"Time elapsed: {round(end_time-start_time, 4)} \n"
        )
        del images, label, image_props

    def preprocess_test_subject(self, subject_id):
        subject_id = subject_id.split(os.extsep, 1)[0]
        escaped_subject_id = re.escape(subject_id)

        imagepaths = [
            impath
            for impath in self.test_imagepaths
            # Check if impath is a modality of subject_id (subject_id + _XXX + .) where XXX are three digits
            if re.search(escaped_subject_id + "_" + r"\d{3}" + ".", os.path.split(impath)[-1])
        ]

        self.sanity_check_modalities_and_return_missing(
            imagepaths=imagepaths,
            normalization_schemes=self.plans["normalization_scheme"],
            allow_missing_modalities=self.allow_missing_modalities,
        )

        images, image_props = self.preprocess_case_for_inference(
            images=imagepaths, sliding_window_prediction=self.sliding_window_prediction
        )
        save_path = join(self.test_target_dir, subject_id)
        torch.save(images, save_path + ".pt")
        save_pickle(image_props, save_path + ".pkl")

    def _preprocess_train_subject(self, subject_id, label_exists: bool, preprocess_label: bool):
        image_props = {}

        # First find relevant images by their paths and save them in the image property pickle
        # Then load them as images
        # The '_' in the end is to avoid treating Case_4_000 AND Case_42_000 as different versions
        # of the label named Case_4 as both would start with "Case_4", however only the correct one is
        # followed by an underscore
        escaped_subject_id = re.escape(subject_id)
        # path to all modalities of subject_id
        imagepaths = [
            impath
            for impath in self.imagepaths
            # Check if impath is a modality of subject_id (subject_id + _XXX + .) where XXX are three digits
            if re.search(escaped_subject_id + "_" + r"\d{3}" + ".", os.path.split(impath)[-1])
        ]

        missing_modalities = self.sanity_check_modalities_and_return_missing(
            imagepaths=imagepaths,
            normalization_schemes=self.plans["normalization_scheme"],
            allow_missing_modalities=self.allow_missing_modalities,
        )

        image_props["image files"] = imagepaths
        images = [read_file_to_nifti_or_np(image) for image in imagepaths]

        if label_exists:
            # Do the same with label
            label = [
                labelpath
                for labelpath in subfiles(join(self.input_dir, "labelsTr"))
                if os.path.split(labelpath)[-1].startswith(subject_id + ".")
            ]
            assert len(label) == 1, f"unexpected number of labels found. Expected 1 and found {len(label)}"
            image_props["label file"] = label[0]
            label = read_file_to_nifti_or_np(label[0], dtype=np.uint8)
        else:
            label = None

        if not self.disable_sanity_checks:
            if label_exists and preprocess_label:
                self.run_sanity_checks(images, label, subject_id, imagepaths)
            else:
                self.run_sanity_checks(images, None, subject_id, imagepaths)

        if label_exists and preprocess_label:
            images, label, image_props = preprocess_case_for_training_with_label(
                images=images,
                label=label,
                normalization_operation=self.plans["normalization_scheme"],
                allow_missing_modalities=self.allow_missing_modalities,
                background_pixel_value=self.background_value_for_first_modality,
                enable_cc_analysis=self.enable_cc_analysis,
                foreground_locs_per_label=self.get_foreground_locs_per_label,
                missing_modality_idxs=missing_modalities,
                crop_to_nonzero=self.plans["crop_to_nonzero"],
                keep_aspect_ratio_when_using_target_size=self.plans["keep_aspect_ratio_when_using_target_size"],
                image_properties=image_props,
                intensities=self.intensities,
                target_orientation=self.plans["target_coordinate_system"],
                target_size=self.target_size,
                target_spacing=self.target_spacing,
                transpose=self.transpose_forward,
            )
            self.verify_label_validity(label, subject_id)

        else:
            images, image_props = preprocess_case_for_training_without_label(
                images=images,
                normalization_operation=self.plans["normalization_scheme"],
                allow_missing_modalities=self.allow_missing_modalities,
                background_pixel_value=self.background_value_for_first_modality,
                missing_modality_idxs=missing_modalities,
                crop_to_nonzero=self.plans["crop_to_nonzero"],
                keep_aspect_ratio_when_using_target_size=self.plans["keep_aspect_ratio_when_using_target_size"],
                image_properties=image_props,
                intensities=self.intensities,
                target_orientation=self.plans["target_coordinate_system"],
                target_size=self.target_size,
                target_spacing=self.target_spacing,
                transpose=self.transpose_forward,
            )
        return images, label, image_props

    def preprocess_case_for_inference(
        self, images: list | tuple, patch_size: tuple = None, ext: str = ".nii.gz", sliding_window_prediction: bool = True
    ):
        """
        Will reorient ONLY if we have valid qform or sform codes.
        with coded=True the methods will return {affine or None} and {0 or 1}.
        If both are 0 we cannot rely on headers for orientations and will
        instead assume images are in the desired orientation already.

        Afterwards images will be normalized and transposed as specified by the
        plans file also used in training.

        Finally images are resampled to the required spacing/size and returned
        as torch tensors of the required shape (b, c, x, y, (z))
        """
        assert isinstance(images, (list, tuple)), "image(s) should be a list or tuple, even if only one " "image is passed"
        self.initialize_properties()

        images = [
            read_file_to_nifti_or_np(image[0]) if isinstance(image, tuple) else read_file_to_nifti_or_np(image)
            for image in images
        ]

        if patch_size is None:
            patch_size = (0,) * len(images[0].shape)

        if sliding_window_prediction is False:
            self.target_size = patch_size

        images, image_properties = preprocess_case_for_inference(
            crop_to_nonzero=self.plans["crop_to_nonzero"],
            keep_aspect_ratio=self.plans["keep_aspect_ratio_when_using_target_size"],
            images=images,
            intensities=self.intensities,
            normalization_scheme=self.plans["normalization_scheme"],
            ext=ext,
            patch_size=patch_size,
            target_size=self.target_size,
            target_spacing=self.target_spacing,
            target_orientation=self.plans["target_coordinate_system"],
            transpose_forward=self.transpose_forward,
            allow_missing_modalities=self.allow_missing_modalities,
        )
        return images, image_properties

    def reverse_preprocessing(self, images: torch.Tensor, image_properties: dict, num_classes: Optional[int] = None):
        """
        Expected shape of images are:
        (b, c, x, y(, z))

        (1) Initialization: Extract relevant properties from the image_properties dictionary.
        (2) Padding Reversion: Reverse the padding applied during preprocessing.
        (3) Resampling and Transposition Reversion: Resize the images to revert the resampling operation.
        Transpose the images back to the original orientation.
        (4) Cropping Reversion (Optional): If cropping to the nonzero bounding box was applied, revert the cropping operation.
        (5) Return: Return the reverted images as a NumPy array.
        The original orientation of the image will be re-applied when saving the prediction
        """
        if num_classes is None:
            num_classes = max(1, len(self.plans["dataset_properties"]["classes"]))

        images, image_properties = reverse_preprocessing(
            crop_to_nonzero=self.plans["crop_to_nonzero"],
            images=images,
            image_properties=image_properties,
            n_classes=num_classes,
            transpose_forward=self.transpose_forward,
            transpose_backward=self.transpose_backward,
        )

        return images, image_properties

    @staticmethod
    def load_plans(plans_path):
        if os.path.splitext(plans_path)[-1] == ".json":
            return load_json(plans_path)
        if os.path.splitext(plans_path)[-1] == ".yaml":
            return load_yaml(plans_path)["config"]["plans"]
        else:
            raise FileNotFoundError(
                f"Plan file not found. Got {plans_path} with ext {os.path.splitext(plans_path)[-1]}. Expects either a '.json' or '.yaml' file."
            )

    def run_sanity_checks(self, images, label, subject_id, imagepaths):
        self.sanity_check_standard_formats(images, label, subject_id, imagepaths)

        if isinstance(images[0], nib.Nifti1Image):
            self.sanity_check_niftis(images, label, subject_id)

    def sanity_check_standard_formats(self, images, label, subject_id, imagepaths):
        assert len(images) > 0, f"found no images for {subject_id + '_'}, " f"attempted imagepaths: {imagepaths}"

        assert (
            len(images[0].shape) == self.plans["dataset_properties"]["data_dimensions"]
        ), f"image should be shape (x, y(, z)) but is {images[0].shape}"

        if label is not None:
            verify_array_shape_is_equal(reference=images[0], target=label, id=subject_id)

        # Make sure all modalities are correctly registered
        if len(images) > 1:
            for image in images:
                verify_array_shape_is_equal(reference=images[0], target=image, id=subject_id)

    @staticmethod
    def sanity_check_niftis(images, label, subject_id):
        if label is not None:
            verify_spacing_is_equal(reference=images[0], target=label, id=subject_id)
            verify_orientation_is_equal(reference=images[0], target=label, id=subject_id)

        if len(images) > 1:
            for image in images:
                verify_spacing_is_equal(reference=images[0], target=image, id=subject_id)
                verify_orientation_is_equal(reference=images[0], target=image, id=subject_id)

    @staticmethod
    def sanity_check_modalities_and_return_missing(imagepaths, normalization_schemes, allow_missing_modalities):
        expected_modalities = set([f"{i:03}" for i in range(len(normalization_schemes))])
        found_modalities = [os.path.split(impath)[-1].split(os.extsep, 1)[0][-3:] for impath in imagepaths]
        missing_modalities = [int(missing_mod) for missing_mod in list(expected_modalities.difference(found_modalities))]

        assert len(imagepaths) > 0, "found no images"
        if not allow_missing_modalities:
            assert not len(missing_modalities) > 0, "found missing modalities and allow_missing_modalities is not enabled."
        return missing_modalities

    def cast_to_numpy_array(self, images: list, label=None, classification=False):
        if label is None and not self.allow_missing_modalities:  # self-supervised
            images = np.array(images, dtype=np.float32)
        elif label is None and self.allow_missing_modalities:  # self-supervised with missing mods
            images = np.array(images, dtype="object")
        elif classification:  # Classification is always "object"
            images.append(label)
            images = np.array(images, dtype="object")
        elif self.allow_missing_modalities:  # segmentation with missing modalities
            images.append(np.array(label)[np.newaxis])
            images = np.array(images, dtype="object")
        else:  # Standard segmentation
            images = np.vstack((np.array(images), np.array(label)[np.newaxis]), dtype=np.float32)
        return images

    def verify_label_validity(self, label, subject_id):
        # Check if the ground truth only contains expected values
        expected_labels = np.array(self.plans["dataset_properties"]["classes"], dtype=np.float32)
        actual_labels = np.unique(label).astype(np.float32)
        verify_labels_are_equal(expected_labels=expected_labels, actual_labels=actual_labels, id=subject_id)

    @staticmethod
    def verify_compression_level(directory: str, compress: bool):
        if compress:
            assert (
                len(subfiles(directory, suffix=".npy")) == 0
            ), "detected uncompressed images in folder while compression is enabled. Please delete uncompressed images first to avoid duplicates"
        else:
            assert (
                len(subfiles(directory, suffix=".npz")) == 0
            ), "detected compressed images in folder while compression is disabled. Please delete compressed images first to avoid duplicates"
