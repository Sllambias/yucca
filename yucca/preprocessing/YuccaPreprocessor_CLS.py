"""
Takes raw data conforming with Yucca standards and preprocesses according to the generic scheme
"""
import numpy as np
import torch
import nibabel as nib
import os
from yucca.preprocessing.YuccaPreprocessor import YuccaPreprocessor
from yucca.paths import yucca_preprocessed_data, yucca_raw_data
from yucca.preprocessing.normalization import normalizer
from yuccalib.utils.nib_utils import get_nib_spacing, get_nib_orientation, reorient_nib_image
from yuccalib.utils.type_conversions import nib_to_np, read_any_file
from yuccalib.image_processing.objects.BoundingBox import get_bbox_for_foreground
from yuccalib.image_processing.cropping_and_padding import crop_to_box, pad_to_size
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


class YuccaPreprocessor_CLS(YuccaPreprocessor):
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
        # of the seg named Case_4 as both would start with "Case_4", however only the correct one is
        # followed by an underscore
        imagepaths = [impath for impath in self.imagepaths if os.path.split(impath)[-1].startswith(subject_id + "_")]

        image_props["image files"] = imagepaths
        images = [read_any_file(image) for image in imagepaths]

        # Do the same with segmentation
        seg = [
            segpath
            for segpath in subfiles(join(self.input_dir, "labelsTr"))
            if os.path.split(segpath)[-1].startswith(subject_id + ".")
        ]
        assert len(seg) < 2, f"unexpected number of segmentations found. Expected 1 or 0 and found {len(seg)}"
        image_props["segmentation file"] = seg[0]
        seg = read_any_file(seg[0], dtype=np.uint8)

        if not self.disable_unittests:
            assert len(images) > 0, f"found no images for {subject_id + '_'}, " f"attempted imagepaths: {imagepaths}"
            assert (
                len(images[0].shape) == self.plans["dataset_properties"]["data_dimensions"]
            ), f"image should be shape (x, y(, z)) but is {images[0].shape}"

            # Make sure all modalities are correctly registered
            if len(images) > 1:
                for image in images:
                    assert images[0].shape == image.shape, (
                        f"Sizes do not match for {subject_id}" f"One is: {images[0].shape} while another is {image.shape}"
                    )

        original_size = np.array(images[0].shape)

        # If qform and sform are both missing the header is corrupt and we do not trust the
        # direction from the affine
        # Make sure you know what you're doing
        valid_header = False
        if isinstance(images[0], nib.Nifti1Image):
            if images[0].get_qform(coded=True)[1] or images[0].get_sform(coded=True)[1]:
                valid_header = True

        if valid_header:
            original_spacing = get_nib_spacing(images[0])
            original_orientation = get_nib_orientation(images[0])
            final_direction = self.plans["target_coordinate_system"]
            images = [nib_to_np(reorient_nib_image(image, original_orientation, final_direction)) for image in images]
            if isinstance(seg, nib.Nifti1Image):
                seg = nib_to_np(reorient_nib_image(seg, original_orientation, final_direction))
        else:
            original_spacing = np.array([1.0] * len(original_size))
            original_orientation = "INVALID"
            final_direction = "INVALID"
            images = [nib_to_np(image) for image in images]
            seg = nib_to_np(seg)

        if self.target_spacing.size:
            target_spacing = self.target_spacing
        else:
            target_spacing = original_spacing

        # Cropping is performed to save computational resources. We are only removing background.
        if self.plans["crop_to_nonzero"]:
            nonzero_box = get_bbox_for_foreground(images[0], background_label=0)
            image_props["crop_to_nonzero"] = nonzero_box
            for i in range(len(images)):
                images[i] = crop_to_box(images[i], nonzero_box)
        else:
            image_props["crop_to_nonzero"] = self.plans["crop_to_nonzero"]

        images = self._resample_and_normalize_case(
            images, None, self.plans["normalization_scheme"], self.transpose_forward, original_spacing, target_spacing
        )

        images = np.array((np.array(images).T, seg), dtype="object")
        images[0] = images[0].T
        final_size = list(images[0][0].shape)

        # For classification there's no foreground classes
        # And no connected components to analyze.
        foreground_locs = []
        numbered_ground_truth = ground_truth_numb_lesion = object_sizes = 0

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
        self, images: list, seg: np.ndarray = None, norm_op=None, transpose=None, original_spacing=None, target_spacing=None
    ):
        # Normalize and Transpose images to target view.
        # Transpose segmentations to target view.
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
        if seg is not None:
            seg = seg.transpose(transpose)
            try:
                seg = resize(seg, output_shape=target_shape, order=0, anti_aliasing=False)
            except OverflowError:
                print("Unexpected values in either shape or seg for resize")
            return images, seg

        return images

    def preprocess_case_for_inference(self, images: list, patch_size: tuple):
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
        assert isinstance(images, list), "image(s) should be a list, even if only one " "image is passed"
        self.initialize_properties()
        image_properties = {}
        images = [nib.load(image) for image in images]

        image_properties["original_spacing"] = get_nib_spacing(images[0])
        image_properties["original_shape"] = np.array(images[0].shape)
        image_properties["qform"] = images[0].get_qform()
        image_properties["sform"] = images[0].get_sform()

        assert len(image_properties["original_shape"]) in [2, 3], "images must be either 2D or 3D for preprocessing"

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
        else:
            print("Insufficient header information. Reorientation will not be attempted.")
            image_properties["reoriented"] = False

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
