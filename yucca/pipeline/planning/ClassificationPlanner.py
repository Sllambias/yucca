from yucca.pipeline.planning.YuccaPlanner import YuccaPlanner
from yucca.pipeline.planning.YuccaPlannerV2 import YuccaPlannerV2


class ClassificationPlanner(YuccaPlanner):
    def __init__(self, task, preprocessor=None, threads=None, disable_unittests=False, view=None):
        super().__init__(task, preprocessor, threads, disable_unittests, view)
        self.name = str(self.__class__.__name__) + str(view or "")
        self.preprocessor = "ClassificationPreprocessor"


class Classification_PsyBrain(YuccaPlannerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = str(self.__class__.__name__)
        self.preprocessor = "ClassificationPreprocessor"
        self.keep_aspect_ratio_when_using_target_size = True
        self.crop_to_nonzero = False

    def determine_target_size_from_fixed_size_or_spacing(self):
        self.fixed_target_size = (192, 224, 192)
        self.fixed_target_spacing = None

    def determine_transpose(self):
        self.transpose_fw = [0, 1, 2]
        self.transpose_bw = [0, 1, 2]


class Classification_PsyBrain128(YuccaPlannerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = str(self.__class__.__name__)
        self.preprocessor = "ClassificationPreprocessor"
        self.keep_aspect_ratio_when_using_target_size = False
        self.crop_to_nonzero = False

    def determine_target_size_from_fixed_size_or_spacing(self):
        self.fixed_target_size = (128, 128, 128)
        self.fixed_target_spacing = None

    def determine_transpose(self):
        self.transpose_fw = [0, 1, 2]
        self.transpose_bw = [0, 1, 2]


class Classification_PsyBrain128V2(Classification_PsyBrain128):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def determine_norm_op_per_modality(self):
        self.norm_op_per_modality = ["volume_wise_znorm", "no_norm"]


class Classification_PsyBrain128V2Cov(Classification_PsyBrain128):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessor = "ClassificationPreprocessorWithCovariates"

    def determine_norm_op_per_modality(self):
        self.norm_op_per_modality = ["volume_wise_znorm", "no_norm"]

class Classification_PsyBrain128V3Cov(Classification_PsyBrain128):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessor = "ClassificationPreprocessorWithCovariates"

    def determine_norm_op_per_modality(self):
        self.norm_op_per_modality = ["volume_wise_znorm"]