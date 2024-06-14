from yucca.pipeline.planning.YuccaPlanner import YuccaPlanner
import numpy as np


class YuccaPlannerV3(YuccaPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = str(self.__class__.__name__)
        self.get_foreground_locations_per_label = True
        self.norm_op = "ct"

    def determine_transpose(self):
        # If no specific view is determined in run training, we select the optimal.
        # This will be the optimal solution in most cases that are not 2.5D training.
        median_spacing = self.dataset_properties["original_median_spacing"]
        sorting_key = np.argsort(median_spacing)[::-1]
        self.transpose_fw = sorting_key.tolist()
        self.transpose_bw = sorting_key.argsort().tolist()
