import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.functional as F
import nibabel as nib
import os
import cc3d
from yuccalib.utils.files_and_folders import load_yaml
from yuccalib.image_processing.objects.BoundingBox import get_bbox_for_foreground
from yuccalib.image_processing.cropping_and_padding import crop_to_box, pad_to_size
from yuccalib.utils.nib_utils import (
    get_nib_spacing,
    get_nib_orientation,
    reorient_nib_image,
)
from yuccalib.utils.type_conversions import nib_to_np, read_any_file
from yucca.paths import yucca_preprocessed_data, yucca_raw_data
from yucca.preprocessing.normalization import normalizer
from multiprocessing import Pool
from skimage.transform import resize
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    subfiles,
    save_pickle,
    maybe_mkdir_p,
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

    def __init__(self, plans_path, task=None, threads=12, disable_unittests=False):
        self.name = str(self.__class__.__name__)
        self.task = task
        self.plans_path = plans_path
        self.plans = self.load_plans(plans_path)
        self.threads = threads
        self.disable_unittests = disable_unittests

        # lists for information we would like to attain
        self.transpose_forward = []
        self.transpose_backward = []
        self.target_spacing = []

    def initialize_paths(self):
        self.target_dir = join(yucca_preprocessed_data, self.task, self.plans["plans_name"])
        self.input_dir = join(yucca_raw_data, self.task)
        self.imagepaths = subfiles(join(self.input_dir, "imagesTr"), suffix=self.image_extension)
        self.subject_ids = subfiles(join(self.input_dir, "labelsTr"), join=False)

    def initialize_properties(self):
        """
        here we basically set up things that are needed for preprocessing during training,
        but that aren't necessary during inference
        """
        self.dataset_properties = self.plans["dataset_properties"]
        self.intensities = self.dataset_properties["intensities"]
        self.image_extension = self.dataset_properties.get("image_extension") or "nii.gz"

        # op values
        self.transpose_forward = np.array(self.plans["transpose_forward"], dtype=int)
        self.transpose_backward = np.array(self.plans["transpose_backward"], dtype=int)
        self.target_spacing = np.array(self.plans["target_spacing"], dtype=float)

    @staticmethod
    def load_plans(plans_path):
        if os.path.splitext(plans_path)[-1] == ".json":
            return load_json(plans_path)
        if os.path.splitext(plans_path)[-1] == ".yaml":
            return load_yaml(plans_path)["config"]["plans"]
        else:
            print(
                f"Plan format not recognized. Got {plans_path} with ext {os.path.splitext(plans_path)[-1]} and expected either a '.json' or '.yaml' file"
            )

    def run(self):
        self.initialize_properties()
        self.initialize_paths()
        maybe_mkdir_p(self.target_dir)

        print(
            f"{'Preprocessing Task:':25.25} {self.task} \n"
            f"{'Using Planner:':25.25} {self.plans_path} \n"
            f"{'Crop to nonzero:':25.25} {self.plans['crop_to_nonzero']} \n"
            f"{'Normalization scheme:':25.25} {self.plans['normalization_scheme']} \n"
            f"{'Transpose Forward:':25.25} {self.transpose_forward} \n"
            f"{'Transpose Backward:':25.25} {self.transpose_backward} \n"
        )
        p = Pool(self.threads)
        p.map(self._preprocess_train_subject, self.subject_ids)
        p.close()
        p.join()

    def _preprocess_train_subject(self, subject_id):
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
        images = [nib.load(image) for image in imagepaths]

        # Do the same with label
        label = join(self.input_dir, "labelsTr", subject_id + ".nii.gz")
        image_props["label file"] = label
        label = nib.load(label)

        if not self.disable_unittests:
            assert len(images) > 0, f"found no images for {subject_id + '_'}, " f"attempted imagepaths: {imagepaths}"

            assert (
                len(images[0].shape) == self.plans["dataset_properties"]["data_dimensions"]
            ), f"image should be shape (x, y(, z)) but is {images[0].shape}"

            # make sure images and labels are correctly registered
            assert images[0].shape == label.shape, (
                f"Sizes do not match for {subject_id}" f"Image is: {images[0].shape} while the label is {label.shape}"
            )

            assert np.allclose(get_nib_spacing(images[0]), get_nib_spacing(label)), (
                f"Spacings do not match for {subject_id}"
                f"Image is: {get_nib_spacing(images[0])} while the label is {get_nib_spacing(label)}"
            )

            assert get_nib_orientation(images[0]) == get_nib_orientation(label), (
                f"Directions do not match for {subject_id}"
                f"Image is: {get_nib_orientation(images[0])} while the label is {get_nib_orientation(label)}"
            )

            # Make sure all modalities are correctly registered
            if len(images) > 1:
                for image in images:
                    assert images[0].shape == image.shape, (
                        f"Sizes do not match for {subject_id}" f"One is: {images[0].shape} while another is {image.shape}"
                    )

                    assert np.allclose(get_nib_spacing(images[0]), get_nib_spacing(image)), (
                        f"Spacings do not match for {subject_id}"
                        f"One is: {get_nib_spacing(images[0])} while another is {get_nib_spacing(image)}"
                    )

                    assert get_nib_orientation(images[0]) == get_nib_orientation(image), (
                        f"Directions do not match for {subject_id}"
                        f"One is: {get_nib_orientation(images[0])} while another is {get_nib_orientation(image)}"
                    )

        original_spacing = get_nib_spacing(images[0])
        original_size = np.array(images[0].shape)

        if self.target_spacing.size:
            target_spacing = self.target_spacing
        else:
            target_spacing = original_spacing

        # If qform and sform are both missing the header is corrupt and we do not trust the
        # direction from the affine
        # Make sure you know what you're doing
        if images[0].get_qform(coded=True)[1] or images[0].get_sform(coded=True)[1]:
            original_orientation = get_nib_orientation(images[0])
            final_direction = self.plans["target_coordinate_system"]
            images = [nib_to_np(reorient_nib_image(image, original_orientation, final_direction)) for image in images]
            label = nib_to_np(reorient_nib_image(label, original_orientation, final_direction))
        else:
            original_orientation = "INVALID"
            final_direction = "INVALID"
            images = [nib_to_np(image) for image in images]
            label = nib_to_np(label)

        # Check if the ground truth only contains expected values
        expected_labels = np.array(self.plans["dataset_properties"]["classes"], dtype=np.float32)
        actual_labels = np.unique(label).astype(np.float32)
        assert np.all(np.isin(actual_labels, expected_labels)), (
            f"Unexpected labels found for {subject_id} \n" f"expected: {expected_labels} \n" f"found: {actual_labels}"
        )

        # Cropping is performed to save computational resources. We are only removing background.
        if self.plans["crop_to_nonzero"]:
            nonzero_box = get_bbox_for_foreground(images[0], background_label=0)
            image_props["crop_to_nonzero"] = nonzero_box
            for i in range(len(images)):
                images[i] = crop_to_box(images[i], nonzero_box)
            label = crop_to_box(label, nonzero_box)
        else:
            image_props["crop_to_nonzero"] = self.plans["crop_to_nonzero"]

        images, label = self._resample_and_normalize_case(
            images,
            label,
            self.plans["normalization_scheme"],
            self.transpose_forward,
            original_spacing,
            target_spacing,
        )

        # Stack and fix dimensions
        images = np.vstack((np.array(images), np.array(label)[np.newaxis]))

        # now AFTER transposition etc., we get some (no need to get all)
        # locations of foreground, that we will later use in the
        # oversampling of foreground classes
        foreground_locs = np.array(np.nonzero(images[-1])).T[::10]
        numbered_ground_truth, ground_truth_numb_lesion = cc3d.connected_components(images[-1], connectivity=26, return_N=True)
        if ground_truth_numb_lesion == 0:
            object_sizes = 0
        else:
            object_sizes = [i * np.prod(target_spacing) for i in np.unique(numbered_ground_truth, return_counts=True)[-1][1:]]

        final_size = list(images[0].shape)

        # save relevant values
        image_props["original_spacing"] = original_spacing
        image_props["original_size"] = original_size
        image_props["original_orientation"] = original_orientation
        image_props["new_spacing"] = target_spacing[self.transpose_forward].tolist()
        image_props["new_size"] = final_size
        image_props["new_direction"] = final_direction
        image_props["foreground_locations"] = foreground_locs
        image_props["n_cc"] = ground_truth_numb_lesion
        image_props["size_cc"] = object_sizes

        print(
            f"size before: {original_size} size after: {image_props['new_size']} \n"
            f"spacing before: {original_spacing} spacing after: {image_props['new_spacing']} \n"
            f"Saving {subject_id} in {arraypath} \n"
        )

        # save the image
        np.save(arraypath, images)

        # save metadata as .pkl
        save_pickle(image_props, picklepath)

    def _resample_and_normalize_case(
        self,
        images: list,
        label: np.ndarray = None,
        norm_op=None,
        transpose=None,
        original_spacing=None,
        target_spacing=None,
    ):
        # Normalize and Transpose images to target view.
        # Transpose labels to target view.
        assert len(images) == len(norm_op) == len(self.intensities), (
            "number of images, "
            "normalization  operations and intensities does not match. \n"
            f"len(images) == {len(images)} \n"
            f"len(norm_op) == {len(norm_op)} \n"
            f"len(self.intensities) == {len(self.intensities)} \n"
        )

        for i in range(len(images)):
            images[i] = normalizer(images[i], scheme=norm_op[i], intensities=self.intensities[i])
            assert len(images[i].shape) == len(transpose), (
                "image and transpose axes do not match. \n"
                f"images[i].shape == {images[i].shape} \n"
                f"transpose == {transpose} \n"
                f"len(images[i].shape) == {len(images[i]).shape} \n"
                f"len(transpose) == {len(transpose)} \n"
            )
            images[i] = images[i].transpose(transpose)
        print(f"Normalized with: {norm_op[0]} \n" f"Transposed with: {transpose}")

        shape_t = images[0].shape
        original_spacing_t = original_spacing[transpose]
        target_spacing_t = target_spacing[transpose]

        # Find new shape based on the target spacing
        target_shape = np.round((original_spacing_t / target_spacing_t).astype(float) * shape_t).astype(int)

        # Resample to target shape and spacing
        for i in range(len(images)):
            try:
                images[i] = resize(images[i], output_shape=target_shape, order=3)
            except OverflowError:
                print("Unexpected values in either shape or image for resize")
        if label is not None:
            label = label.transpose(transpose)
            try:
                label = resize(label, output_shape=target_shape, order=0, anti_aliasing=False)
            except OverflowError:
                print("Unexpected values in either shape or label for resize")
            return images, label

        return images

    def preprocess_case_for_inference(self, images: list | tuple, patch_size: tuple):
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
        image_properties = {}
        ext = images[0][0].split(os.extsep, 1)[1] if isinstance(images[0], tuple) else images[0].split(os.extsep, 1)[1]
        images = [read_any_file(image[0]) if isinstance(image, tuple) else read_any_file(image) for image in images]

        image_properties["image_extension"] = ext
        image_properties["original_shape"] = np.array(images[0].shape)

        assert len(image_properties["original_shape"]) in [
            2,
            3,
        ], "images must be either 2D or 3D for preprocessing"

        image_properties["original_spacing"] = np.array([1.0] * len(image_properties["original_shape"]))
        image_properties["qform"] = None
        image_properties["sform"] = None
        image_properties["reoriented"] = False
        image_properties["affine"] = None

        if isinstance(images[0], nib.Nifti1Image):
            image_properties["original_spacing"] = get_nib_spacing(images[0])
            image_properties["qform"] = images[0].get_qform()
            image_properties["sform"] = images[0].get_sform()
            # Check if header is valid and then attempt to orient to target orientation.
            if (
                images[0].get_qform(coded=True)[1]
                or images[0].get_sform(coded=True)[1]
                and self.plans.get("target_coordinate_system")
            ):
                image_properties["reoriented"] = True
                original_orientation = get_nib_orientation(images[0])
                image_properties["original_orientation"] = original_orientation
                images = [
                    reorient_nib_image(image, original_orientation, self.plans["target_coordinate_system"]) for image in images
                ]
                image_properties["new_orientation"] = get_nib_orientation(images[0])
            image_properties["affine"] = images[0].affine

        images = [nib_to_np(image) for image in images]

        image_properties["uncropped_shape"] = np.array(images[0].shape)

        if self.plans["crop_to_nonzero"]:
            nonzero_box = get_bbox_for_foreground(images[0], background_label=0)
            for i in range(len(images)):
                images[i] = crop_to_box(images[i], nonzero_box)
            image_properties["nonzero_box"] = nonzero_box

        image_properties["cropped_shape"] = np.array(images[0].shape)

        images = self._resample_and_normalize_case(
            images,
            norm_op=self.plans["normalization_scheme"],
            transpose=self.transpose_forward,
            original_spacing=image_properties["original_spacing"],
            target_spacing=self.target_spacing,
        )

        # From this point images are shape (1, c, x, y, z)
        image_properties["resampled_transposed_shape"] = np.array(images[0].shape)

        for i in range(len(images)):
            images[i], padding = pad_to_size(images[i], patch_size)
        image_properties["padded_shape"] = np.array(images[0].shape)
        image_properties["padding"] = padding

        # Stack and fix dimensions
        images = np.stack(images)[np.newaxis]

        return torch.tensor(images, dtype=torch.float32), image_properties

    def reverse_preprocessing(self, images: torch.Tensor, image_properties: dict):
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
        image_properties["save_format"] = image_properties.get("image_extension")
        nclasses = max(1, len(self.plans["dataset_properties"]["classes"]))
        canvas = torch.zeros((1, nclasses, *image_properties["uncropped_shape"]), dtype=images.dtype)
        shape_after_crop = image_properties["cropped_shape"]
        shape_after_crop_transposed = shape_after_crop[self.transpose_forward]
        pad = image_properties["padding"]

        assert np.all(images.shape[2:] == image_properties["padded_shape"]), (
            f"Reversing padding: "
            f"image should be of shape: {image_properties['padded_shape']}"
            f"but is: {images.shape[2:]}"
        )
        shape = images.shape[2:]
        if len(pad) == 6:
            images = images[
                :,
                :,
                pad[0] : shape[0] - pad[1],
                pad[2] : shape[1] - pad[3],
                pad[4] : shape[2] - pad[5],
            ]
        elif len(pad) == 4:
            images = images[:, :, pad[0] : shape[0] - pad[1], pad[2] : shape[1] - pad[3]]

        assert np.all(images.shape[2:] == image_properties["resampled_transposed_shape"]), (
            f"Reversing resampling and tranposition: "
            f"image should be of shape: {image_properties['resampled_transposed_shape']}"
            f"but is: {images.shape[2:]}"
        )
        # Here we Interpolate the array to the original size. The shape starts as [H, W (,D)]. For Torch functionality it is changed to [B, C, H, W (,D)].
        # Afterwards it's squeezed back into [H, W (,D)] and transposed to the original direction.
        images = F.interpolate(images, size=shape_after_crop_transposed.tolist(), mode="trilinear").permute(
            [0, 1] + [i + 2 for i in self.transpose_backward]
        )

        assert np.all(images.shape[2:] == image_properties["cropped_shape"]), (
            f"Reversing cropping: "
            f"image should be of shape: {image_properties['cropped_shape']}"
            f"but is: {images.shape[2:]}"
        )
        assert np.all(images.shape[2:] == image_properties["resampled_transposed_shape"]), (
            f"Reversing resampling and tranposition: "
            f"image should be of shape: {image_properties['resampled_transposed_shape']}"
            f"but is: {images.shape[2:]}"
        )
        # Here we Interpolate the array to the original size. The shape starts as [H, W (,D)]. For Torch functionality it is changed to [B, C, H, W (,D)].
        # Afterwards it's squeezed back into [H, W (,D)] and transposed to the original direction.
        images = F.interpolate(images, size=shape_after_crop_transposed.tolist(), mode="trilinear").permute(
            [0, 1] + [i + 2 for i in self.transpose_backward]
        )

        assert np.all(images.shape[2:] == image_properties["cropped_shape"]), (
            f"Reversing cropping: "
            f"image should be of shape: {image_properties['cropped_shape']}"
            f"but is: {images.shape[2:]}"
        )

        if self.plans["crop_to_nonzero"]:
            bbox = image_properties["nonzero_box"]
            slices = [
                slice(None),
                slice(None),
                slice(bbox[0], bbox[1]),
                slice(bbox[2], bbox[3]),
            ]
            if len(bbox) == 6:
                slices.append(
                    slice(bbox[4], bbox[5]),
                )
            canvas[slices] = images
        else:
            canvas = images
        return canvas.numpy(), image_properties
