from yucca.pipeline.preprocessing import ClassificationPreprocessor
import yucca
import numpy as np
from yucca.paths import get_preprocessed_data_path, get_raw_data_path
from yucca.pipeline.planning.dataset_properties import create_dataset_properties
from yucca.functional.utils.files_and_folders import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
    isfile,
    load_pickle,
    save_json,
)
from yucca.functional.planning import make_plans_file, add_stats_to_plans_post_preprocessing
from yucca.pipeline.preprocessing import UnsupervisedPreprocessor


class YuccaPlanner(object):
    """
    For subclasses changing parameters: remember to also change the name.
    otherwise different planners will save files with identical names and override eachother.

    target_coordinate_system defines the starting orientation for medical images.
    This is only used if the nifti files contain valid headers.

    transpose_forward is applied AFTER images are set to starting orientation defined by
    self.target_coordinate_system, to train on specific views
    (e.g. 2.5D model ensembles where each 2D model is trained on one of the axial, coronal
    and sagittal views).

    transpose_backward is used to revert any transformation applied by transpose_forward.

    crop_to_nonzero removes all the background outside the bounding box of the object
    (e.g. the brain).

    norm_op defines the normalization operation applied in preprocessing.

    Assuming volumes are RAS, i.e. [sagittal, coronal, axial] == [left-right, back-front, down-up],
    this means 2D models will be trained on sagittal slices with this planner.
    """

    def __init__(
        self,
        task,
        preprocessor="YuccaPreprocessor",
        threads=None,
        disable_cc_analysis=True,
        disable_sanity_checks=False,
        view=None,
        preprocess_test=False,
    ):
        # Required arguments
        self.task = task

        # Planner Specific settings
        self.name = str(self.__class__.__name__) + str(view or "")
        self.compress = False
        self.target_coordinate_system = "RAS"
        self.crop_to_nonzero = True
        self.norm_op = "standardize"
        self.allow_missing_modalities = False
        self.get_foreground_locations_per_label = False

        # This is only relevant for planners with fixed sizes.
        self.keep_aspect_ratio_when_using_target_size = True

        # Don't change the remaining variables unless you know what you're doing
        # Threading speeds up the process. Unittests should by default be enabled.
        self.preprocessor = preprocessor
        self.threads = int(threads) if threads is not None else 1
        self.disable_sanity_checks = disable_sanity_checks
        self.disable_cc_analysis = disable_cc_analysis

        self.plans = {}
        self.suggested_dimensionality = "3D"
        self.view = view
        self.preprocess_test = preprocess_test

    def plan(self):
        self.set_paths()

        if not isfile(join(self.target_dir, "dataset_properties.pkl")):
            print("Properties file not found. Creating one now - this might take a few minutes")
            create_dataset_properties(data_dir=self.in_dir, save_dir=self.target_dir, num_workers=self.threads)
        self.dataset_properties = load_pickle(join(self.target_dir, "dataset_properties.pkl"))

        self.determine_norm_op_per_modality()
        self.determine_transpose()
        self.determine_target_size_from_fixed_size_or_spacing()
        self.determine_task_type()
        self.validate_target_size()
        self.drop_keys_from_dict(dict=self.dataset_properties, keys=[])
        self.populate_plans_file()

        save_json(self.plans, self.plans_path, sort_keys=False)

    def determine_norm_op_per_modality(self):
        self.norm_op_per_modality = [self.norm_op for _ in self.dataset_properties["modalities"]]

    def determine_transpose(self):
        # If no specific view is determined in run training, we select the optimal.
        # This will be the optimal solution in most cases that are not 2.5D training.
        dims = self.dataset_properties["data_dimensions"]

        if dims == 2:
            self.suggested_dimensionality = "2D"

        if not self.view:
            median_size = np.median(self.dataset_properties["original_sizes"], 0)
            sorting_key = median_size.argsort()
            median_size_sorted = median_size[sorting_key]
            median_spacing_sorted = np.median(self.dataset_properties["original_spacings"], 0)[sorting_key]
            if median_size_sorted[0] < median_size_sorted[1] / 2 and median_spacing_sorted[0] > median_spacing_sorted[1] * 2:
                self.transpose_fw = sorting_key.tolist()
                self.transpose_bw = sorting_key.argsort().tolist()
                self.suggested_dimensionality = "2D"
            else:
                self.transpose_fw = [0, 1, 2][:dims]
                self.transpose_bw = [0, 1, 2][:dims]

        if self.view == "X":
            self.suggested_dimensionality = "2D"
            self.transpose_fw = [0, 1, 2][:dims]
            self.transpose_bw = [0, 1, 2][:dims]

        if self.view == "Y":
            self.suggested_dimensionality = "2D"
            self.transpose_fw = [1, 0, 2][:dims]
            self.transpose_bw = [1, 0, 2][:dims]

        if self.view == "Z":
            self.suggested_dimensionality = "2D"
            assert dims == 3, "Z-view does not exist for 2D data"
            self.transpose_fw = [2, 1, 0]
            self.transpose_bw = [2, 1, 0]

        assert self.transpose_fw is not None, "no transposition, something is wrong."

    def validate_target_size(self):
        assert self.fixed_target_size is None or self.fixed_target_spacing is None, (
            "only one of target size or target spacing should be specified"
            f" but both are specified here as {self.fixed_target_size} and {self.fixed_target_spacing} respectively"
        )

    def determine_target_size_from_fixed_size_or_spacing(self):
        self.fixed_target_size = None
        self.fixed_target_spacing = self.dataset_properties["original_median_spacing"]

    def drop_keys_from_dict(self, dict, keys):
        for key in keys:
            dict.pop(key)

    def preprocess(self):
        preprocessor = recursive_find_python_class(
            folder=[join(yucca.__path__[0], "pipeline", "preprocessing")],
            class_name=self.preprocessor,
            current_module="yucca.pipeline.preprocessing",
        )

        preprocessor = preprocessor(
            plans_path=self.plans_path,
            task=self.task,
            threads=self.threads,
            disable_sanity_checks=self.disable_sanity_checks,
            allow_missing_modalities=self.allow_missing_modalities,
            compress=self.compress,
            get_foreground_locs_per_label=self.get_foreground_locations_per_label,
            preprocess_test=self.preprocess_test,
        )
        preprocessor.run()
        self.postprocess()

    def populate_plans_file(self):
        self.plans = make_plans_file(
            allow_missing_modalities=self.allow_missing_modalities,
            crop_to_nonzero=self.crop_to_nonzero,
            classes=self.dataset_properties["classes"],
            norm_op=self.norm_op_per_modality,
            modalities=self.dataset_properties["modalities"],
            plans_name=self.name,
            preprocessor=self.preprocessor,
            dataset_properties=self.dataset_properties,
            keep_aspect_ratio_when_using_target_size=self.keep_aspect_ratio_when_using_target_size,
            task_type=self.task_type,
            target_coordinate_system=self.target_coordinate_system,
            target_spacing=self.fixed_target_spacing,
            target_size=self.fixed_target_size,
            transpose_forward=self.transpose_fw,
            transpose_backward=self.transpose_bw,
            suggested_dimensionality=self.suggested_dimensionality,
        )

    def postprocess(self):
        self.plans = add_stats_to_plans_post_preprocessing(plans=self.plans, directory=self.plans_folder)
        save_json(self.plans, self.plans_path, sort_keys=False)

    def set_paths(self):
        # Setting up paths
        self.in_dir = join(get_raw_data_path(), self.task)
        self.target_dir = join(get_preprocessed_data_path(), self.task)
        self.plans_folder = join(self.target_dir, self.name)
        self.plans_path = join(self.plans_folder, self.name + "_plans.json")
        ensure_dir_exists(join(self.target_dir, self.name))

    def determine_task_type(self):
        preprocessor_class = recursive_find_python_class(
            folder=[join(yucca.__path__[0], "pipeline", "preprocessing")],
            class_name=self.preprocessor,
            current_module="yucca.pipeline.preprocessing",
        )
        # If key is not present in plan then we try to infer the task_type from the Type of Preprocessor
        assert (
            preprocessor_class
        ), f"{self.preprocessor} was found in plans, but no class with the corresponding name was found"
        if issubclass(preprocessor_class, ClassificationPreprocessor):
            self.task_type = "classification"
        elif issubclass(preprocessor_class, UnsupervisedPreprocessor):
            self.task_type = "self-supervised"
        else:
            self.task_type = "segmentation"


class YuccaPlannerX(YuccaPlanner):
    """
    Used to train (mostly 2D) models on the Coronal view of 3D volumes.
    """

    def __init__(
        self,
        task,
        preprocessor="YuccaPreprocessor",
        threads=12,
        disable_cc_analysis=True,
        disable_sanity_checks=False,
        view=None,
    ):
        super().__init__(
            task,
            preprocessor=preprocessor,
            threads=threads,
            disable_cc_analysis=disable_cc_analysis,
            disable_sanity_checks=disable_sanity_checks,
            view=view,
        )
        self.view = "X"


class YuccaPlannerY(YuccaPlanner):
    """
    Used to train (mostly 2D) models on the Coronal view of 3D volumes.
    """

    def __init__(
        self,
        task,
        preprocessor="YuccaPreprocessor",
        threads=None,
        disable_cc_analysis=True,
        disable_sanity_checks=False,
        view=None,
    ):
        super().__init__(
            task,
            preprocessor=preprocessor,
            threads=threads,
            disable_cc_analysis=disable_cc_analysis,
            disable_sanity_checks=disable_sanity_checks,
            view=view,
        )
        self.view = "Y"


class YuccaPlannerZ(YuccaPlanner):
    """
    Used to train (mostly 2D) models on the Axial view of 3D volumes.
    """

    def __init__(
        self,
        task,
        preprocessor="YuccaPreprocessor",
        threads=None,
        disable_cc_analysis=True,
        disable_sanity_checks=False,
        view=None,
    ):
        super().__init__(
            task,
            preprocessor=preprocessor,
            threads=threads,
            disable_cc_analysis=disable_cc_analysis,
            disable_sanity_checks=disable_sanity_checks,
            view=view,
        )
        self.view = "Z"


class UnsupervisedPlanner(YuccaPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = str(self.__class__.__name__)
        self.norm_op = "volume_wise_znorm"
        self.preprocessor = "UnsupervisedPreprocessor"  # hard coded


class YuccaPlannerMinMax(YuccaPlanner):
    """
    Standardizes the images to 0-1 range.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = str(self.__class__.__name__)
        self.norm_op = "minmax"


class YuccaPlanner_224x224_MinMax(YuccaPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = str(self.__class__.__name__)
        self.norm_op = "255to1"

    def determine_target_size_from_fixed_size_or_spacing(self):
        self.fixed_target_size = (224, 224)
        self.fixed_target_spacing = None
