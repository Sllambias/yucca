from yucca.planning.YuccaPlanner import YuccaPlanner


class YuccaPlanner_NoResample(YuccaPlanner):
    def __init__(self, task, view=None):
        super().__init__(task=task, view=view)
        self.name = str(self.__class__.__name__) + str(view or "")

    def determine_target_size_from_fixed_size_or_spacing(self):
        self.fixed_target_spacing = []
        self.fixed_target_size = []
