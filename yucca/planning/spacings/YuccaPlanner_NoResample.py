# This planner can be used for highly anisotropic data.
# It will not resample data to a uniform spacing but rather leave all samples
# with their native spacing.
from yucca.planning.YuccaPlannerV2 import YuccaPlannerV2


class YuccaPlannerV2_NoResample(YuccaPlannerV2):
    def __init__(self, task, threads=2, disable_unittests=False, view=None):
        super().__init__(task, threads, disable_unittests, view)
        self.name = str(self.__class__.__name__) + str(view or '')
        self.view = view

    def determine_spacing(self):
        self.target_spacing = []
