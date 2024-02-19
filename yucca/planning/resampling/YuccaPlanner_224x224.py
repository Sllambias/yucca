from yucca.planning.YuccaPlanner import YuccaPlanner


class YuccaPlanner_224x224(YuccaPlanner):
    def __init__(self, task, preprocessor="YuccaPreprocessor", threads=None, disable_unittests=False, view=None):
        super().__init__(task, preprocessor, threads, disable_unittests, view)
        self.name = str(self.__class__.__name__) + str(view or "")

    def determine_target_size_from_fixed_size_or_spacing(self):
        self.fixed_target_size = [224, 224]
        self.fixed_target_spacing = None
