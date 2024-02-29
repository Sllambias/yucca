from yucca.planning.YuccaPlanner import YuccaPlanner


class ClassificationPlanner(YuccaPlanner):
    def __init__(self, task, preprocessor=None, threads=None, disable_unittests=False, view=None):
        super().__init__(task, preprocessor, threads, disable_unittests, view)
        self.name = str(self.__class__.__name__) + str(view or "")
        self.preprocessor = "ClassificationPreprocessor"
