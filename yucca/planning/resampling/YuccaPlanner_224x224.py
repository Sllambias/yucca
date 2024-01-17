# This planner can be used for highly anisotropic data.
# It will not resample data to a uniform spacing but rather leave all samples
# with their native spacing.
from yucca.planning.YuccaPlanner import YuccaPlanner


class YuccaPlanner_MaxSize(YuccaPlanner):
    def __init__(self, task, preprocessor="YuccaPreprocessor", threads=2, disable_unittests=False, view=None):
        super().__init__(task, preprocessor, threads, disable_unittests, view)
        self.name = str(self.__class__.__name__) + str(view or "")
        self.view = view

    def _determine_target_spacing_or_size(self):
        self.target_size = [224, 224]
        self.target_spacing = None
