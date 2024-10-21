from yucca.pipeline.planning.YuccaPlanner import UnsupervisedPlanner


class UnsupervisedPlannerUnitSpacing(UnsupervisedPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = str(self.__class__.__name__)
        self.preprocessor = "UnsupervisedPreprocessor"  # hard coded

    def determine_target_size_from_fixed_size_or_spacing(self):
        self.fixed_target_spacing = [1.0, 1.0, 1.0]
        self.fixed_target_size = None
