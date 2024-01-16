# This planner can be used for highly anisotropic data.
# It will not resample data to a uniform spacing but rather leave all samples
# with their native spacing.
from yucca.planning.YuccaPlanner import YuccaPlanner


class YuccaPlanner_NoResample(YuccaPlanner):
    def __init__(self, task, view=None):
        super().__init__(task=task, view=view)
        self.name = str(self.__class__.__name__) + str(view or "")
        self.view = view

    def determine_spacing(self):
        self.target_spacing = []
        self.target_size = []
