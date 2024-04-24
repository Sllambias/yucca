from typing import Optional


def make_plans_file(
    dataset_properties: dict,
    crop_to_nonzero: bool = True,
    keep_aspect_ratio_when_using_target_size: bool = True,
    norm_op: str = "standardize",
    preprocessor: str = "YuccaPreprocessor",
    target_coordinate_system: str = "RAS",
    target_spacing: list = [1.0, 1.0, 1.0],
    target_size: Optional[list] = None,
    transpose_forward: list = [0, 1, 2],
    transpose_backward: list = [0, 1, 2],
):
    plans = {}
    plans["target_coordinate_system"] = target_coordinate_system
    # When True all the background between the volumes and the edges of the image is removed
    plans["crop_to_nonzero"] = crop_to_nonzero

    # Change this to have different normalization schemes for all or some modalities
    plans["normalization_scheme"] = [norm_op for _ in self.dataset_properties["modalities"]]

    plans["transpose_forward"] = transpose_forward
    plans["transpose_backward"] = transpose_backward

    # Defaults to the median spacing of the training data.
    # Change the determine_spacing() function to use different spacings
    plans["keep_aspect_ratio_when_using_target_size"] = keep_aspect_ratio_when_using_target_size
    plans["target_size"] = self.fixed_target_size
    plans["target_spacing"] = self.fixed_target_spacing
    plans["preprocessor"] = self.preprocessor
    plans["dataset_properties"] = self.dataset_properties
    plans["plans_name"] = self.name
    plans["suggested_dimensionality"] = self.suggested_dimensionality
    plans["allow_missing_modalities"] = self.allow_missing_modalities
