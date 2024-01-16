# This planner can be used for highly anisotropic data.
# It will not resample data to a uniform spacing but rather leave all samples
# with their native spacing.
from yucca.planning.YuccaPlanner import YuccaPlanner


class YuccaPlanner_UniformSize_Max(YuccaPlanner):
    def __init__(self, task, preprocessor="YuccaPreprocessor", threads=2, disable_unittests=False, view=None):
        super().__init__(task, preprocessor, threads, disable_unittests, view)
        self.name = str(self.__class__.__name__) + str(view or "")
        self.view = view
        self.patch_based_training = False
        self.input_size = "max"

    def determine_spacing(self):
        self.target_spacing = []
