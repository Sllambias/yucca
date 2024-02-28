from yucca.planning.YuccaPlanner import YuccaPlanner


class YuccaPlanner_1_1_1(YuccaPlanner):
    def __init__(self, task, preprocessor="YuccaPreprocessor", threads=None, disable_sanity_checks=False, view=None):
        super().__init__(
            task, preprocessor=preprocessor, threads=threads, disable_sanity_checks=disable_sanity_checks, view=view
        )
        self.name = str(self.__class__.__name__) + str(view or "")
        self.view = view
        self.norm_op = "volume_wise_znorm"

    def determine_target_size_from_fixed_size_or_spacing(self):
        self.fixed_target_spacing = [1, 1, 1]
        self.fixed_target_size = None
