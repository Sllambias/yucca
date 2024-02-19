from yucca.planning.YuccaPlanner import UnsupervisedPlanner


class UnsupervisedPlannerUnitSpacing(UnsupervisedPlanner):
    def __init__(self, task, preprocessor="UnsupervisedPreprocessor", threads=None, disable_sanity_checks=False, view=None):
        super().__init__(
            task, preprocessor=preprocessor, threads=threads, disable_sanity_checks=disable_sanity_checks, view=view
        )
        self.name = str(self.__class__.__name__) + str(view or "")
        self.view = view

    def determine_target_size_from_fixed_size_or_spacing(self):
        self.fixed_target_spacing = [1.0, 1.0, 1.0]
        self.fixed_target_size = None
