# This planner can be used for highly anisotropic data.
# It will not resample data to a uniform spacing but rather leave all samples
# with their native spacing.
from yucca.planning.YuccaPlanner import YuccaPlanner


class YuccaPlanner_1_1_125(YuccaPlanner):
    def __init__(self, task, preprocessor="YuccaPreprocessor", threads=2, disable_sanity_checks=False, view=None):
        super().__init__(
            task, preprocessor=preprocessor, threads=threads, disable_sanity_checks=disable_sanity_checks, view=view
        )
        self.name = str(self.__class__.__name__) + str(view or "")
        self.view = view

    def determine_spacing(self):
        self.target_spacing = [1, 1, 1.25]
