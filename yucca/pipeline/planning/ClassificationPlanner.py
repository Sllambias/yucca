from yucca.pipeline.planning.YuccaPlanner import YuccaPlanner
from yucca.pipeline.planning.YuccaPlannerV2 import YuccaPlannerV2


class ClassificationPlanner(YuccaPlanner):
    def __init__(self, task, preprocessor=None, threads=None, disable_unittests=False, view=None):
        super().__init__(task, preprocessor, threads, disable_unittests, view)
        self.name = str(self.__class__.__name__) + str(view or "")
        self.preprocessor = "ClassificationPreprocessor"


class ClassificationV2_192x256x256(YuccaPlannerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = str(self.__class__.__name__)
        self.preprocessor = "ClassificationPreprocessor"

    def determine_target_size_from_fixed_size_or_spacing(self):
        self.fixed_target_size = (192, 256, 256)
        self.fixed_target_spacing = None


class ClassificationV2_128x128x128(YuccaPlannerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = str(self.__class__.__name__)
        self.preprocessor = "ClassificationPreprocessor"

    def determine_target_size_from_fixed_size_or_spacing(self):
        self.fixed_target_size = (128, 128, 128)
        self.fixed_target_spacing = None


class ClassificationV2_192x192x192(YuccaPlannerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = str(self.__class__.__name__)
        self.preprocessor = "ClassificationPreprocessor"

    def determine_target_size_from_fixed_size_or_spacing(self):
        self.fixed_target_size = (192, 192, 192)
        self.fixed_target_spacing = None
