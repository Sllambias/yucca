from yucca.pipeline.planning.YuccaPlanner import YuccaPlanner


class YuccaPlanner_1_1_125(YuccaPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = str(self.__class__.__name__)

    def determine_target_size_from_fixed_size_or_spacing(self):
        self.fixed_target_spacing = [1, 1, 1.25]
        self.fixed_target_size = None
