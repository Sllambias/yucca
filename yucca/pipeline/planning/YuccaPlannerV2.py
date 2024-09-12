from yucca.pipeline.planning.YuccaPlanner import YuccaPlanner
import numpy as np


class YuccaPlannerV2(YuccaPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = str(self.__class__.__name__)
        self.get_foreground_locations_per_label = True

    def determine_norm_op_per_modality(self):
        self.norm_op_per_modality = [
            "ct" if "ct" in mod.lower() else "volume_wise_znorm" for mod in self.dataset_properties["modalities"].values()
        ]

    def determine_transpose(self):
        # If no specific view is determined in run training, we select the optimal.
        # This will be the optimal solution in most cases that are not 2.5D training.
        median_spacing = self.dataset_properties["original_median_spacing"]
        sorting_key = np.argsort(median_spacing)[::-1]
        self.transpose_fw = sorting_key.tolist()
        self.transpose_bw = sorting_key.argsort().tolist()

    def determine_target_size_from_fixed_size_or_spacing(self):
        # If the dataset contains data with very different spacings this planner will
        # not use the median spacing, but rather the first quartile spacing.
        # By doing so it intentionally avoids excessive downsampling and instead favors upsampling.
        self.fixed_target_size = None
        self.fixed_target_spacing = self.dataset_properties["original_median_spacing"]
        spacing_anisotropy_threshold = 2.5
        spacings = np.array(self.dataset_properties["original_spacings"])
        for dim in range(spacings.shape[1]):
            low = np.percentile(spacings[:, dim], 5)
            high = np.percentile(spacings[:, dim], 95)
            if low * spacing_anisotropy_threshold < high:
                self.fixed_target_spacing[dim] = np.percentile(spacings[:, dim], 10)
