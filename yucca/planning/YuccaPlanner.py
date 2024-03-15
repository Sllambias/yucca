import yucca
import numpy as np
from yucca.paths import yucca_preprocessed_data, yucca_raw_data
from yucca.planning.dataset_properties import create_dataset_properties
from yucca.utils.files_and_folders import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, isfile, load_pickle, save_json, subfiles


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
    ):
        # Required arguments
        self.task = task

        # Planner Specific settings
        self.name = str(self.__class__.__name__) + str(view or "")
        self.target_coordinate_system = "RAS"
        self.crop_to_nonzero = True
        self.norm_op = "standardize"

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

    def plan(self):
        self.set_paths()

        if not isfile(join(self.target_dir, "dataset_properties.pkl")):
            print("Properties file not found. Creating one now - this might take a few minutes")
            create_dataset_properties(data_dir=self.in_dir, save_dir=self.target_dir, num_workers=self.threads)
        self.dataset_properties = load_pickle(join(self.target_dir, "dataset_properties.pkl"))

        self.determine_transpose()
        self.determine_target_size_from_fixed_size_or_spacing()
        self.validate_target_size()
        self.drop_keys_from_dict(dict=self.dataset_properties, keys=[])
        self.populate_plans_file()

        save_json(self.plans, self.plans_path, sort_keys=False)

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
            folder=[join(yucca.__path__[0], "preprocessing")],
            class_name=self.preprocessor,
            current_module="yucca.preprocessing",
        )

        preprocessor = preprocessor(self.plans_path, self.task, self.threads, self.disable_sanity_checks)
        preprocessor.run()
        self.postprocess()

    def populate_plans_file(self):
        self.plans["target_coordinate_system"] = self.target_coordinate_system
        # When True all the background between the volumes and the edges of the image is removed
        self.plans["crop_to_nonzero"] = self.crop_to_nonzero

        # Change this to have different normalization schemes for all or some modalities
        self.plans["normalization_scheme"] = [self.norm_op for _ in self.dataset_properties["modalities"]]

        self.plans["transpose_forward"] = self.transpose_fw
        self.plans["transpose_backward"] = self.transpose_bw

        # Defaults to the median spacing of the training data.
        # Change the determine_spacing() function to use different spacings
        self.plans["keep_aspect_ratio_when_using_target_size"] = self.keep_aspect_ratio_when_using_target_size
        self.plans["target_size"] = self.fixed_target_size
        self.plans["target_spacing"] = self.fixed_target_spacing
        self.plans["preprocessor"] = self.preprocessor
        self.plans["dataset_properties"] = self.dataset_properties
        self.plans["plans_name"] = self.name
        self.plans["suggested_dimensionality"] = self.suggested_dimensionality

    def postprocess(self):
        pkl_files = subfiles(self.plans_folder, suffix=".pkl")

        new_spacings = []
        new_sizes = []
        n_cc = []
        size_cc = []
        for pkl_file in pkl_files:
            pkl_file = load_pickle(pkl_file)
            new_spacings.append(pkl_file["new_spacing"])
            new_sizes.append(pkl_file["new_size"])
            n_cc.append(pkl_file["label_cc_n"])
            if np.mean(pkl_file["label_cc_sizes"]) > 0:
                size_cc.append(np.mean(pkl_file["label_cc_sizes"], dtype=int))

        mean_size = np.mean(new_sizes, 0, dtype=int).tolist()
        min_size = np.min(new_sizes, 0).tolist()
        max_size = np.max(new_sizes, 0).tolist()

        if len(size_cc) > 0:
            mean_cc = np.mean(size_cc, dtype=int).tolist()
            min_cc = np.min(size_cc).tolist()
            max_cc = np.max(size_cc).tolist()
            mean_n_cc = np.mean(n_cc, dtype=int).tolist()
            min_n_cc = np.min(n_cc, initial=-1).tolist()
            max_n_cc = np.max(n_cc, initial=-1).tolist()
        else:
            mean_cc = min_cc = max_cc = mean_n_cc = min_n_cc = max_n_cc = 0

        self.plans["new_sizes"] = new_sizes
        self.plans["new_spacings"] = new_spacings
        self.plans["new_mean_size"] = mean_size
        self.plans["new_min_size"] = min_size
        self.plans["new_max_size"] = max_size
        self.plans["mean_cc_size"] = mean_cc
        self.plans["max_cc_size"] = max_cc
        self.plans["min_cc_size"] = min_cc
        self.plans["mean_n_cc"] = mean_n_cc
        self.plans["max_n_cc"] = max_n_cc
        self.plans["min_n_cc"] = min_n_cc

        save_json(self.plans, self.plans_path, sort_keys=False)

    def set_paths(self):
        # Setting up paths
        self.in_dir = join(yucca_raw_data, self.task)
        self.target_dir = join(yucca_preprocessed_data, self.task)
        self.plans_folder = join(self.target_dir, self.name)
        self.plans_path = join(self.plans_folder, self.name + "_plans.json")
        maybe_mkdir_p(join(self.target_dir, self.name))


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
    def __init__(self, task, preprocessor=None, threads=None, disable_sanity_checks=False, view=None):
        super().__init__(
            task, preprocessor=preprocessor, threads=threads, disable_sanity_checks=disable_sanity_checks, view=view
        )
        self.name = str(self.__class__.__name__)
        self.norm_op = "volume_wise_znorm"
        self.preprocessor = "UnsupervisedPreprocessor"  # hard coded


class YuccaPlannerMinMax(YuccaPlanner):
    """
    Standardizes the images to 0-1 range.
    """

    def __init__(self, task, preprocessor=None, threads=None, disable_sanity_checks=False, view=None):
        super().__init__(
            task, preprocessor=preprocessor, threads=threads, disable_sanity_checks=disable_sanity_checks, view=view
        )
        self.name = str(self.__class__.__name__)
        self.norm_op = "minmax"


class YuccaPlanner_224x224_MinMax(YuccaPlanner):
    def __init__(self, task, preprocessor="YuccaPreprocessor", threads=None, disable_unittests=False, view=None):
        super().__init__(task, preprocessor, threads, disable_unittests, view)
        self.name = str(self.__class__.__name__) + str(view or "")
        self.norm_op = "255to1"

    def determine_target_size_from_fixed_size_or_spacing(self):
        self.fixed_target_size = (224, 224)
        self.fixed_target_spacing = None
