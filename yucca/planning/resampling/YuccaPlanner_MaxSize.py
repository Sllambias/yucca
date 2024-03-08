from yucca.planning.YuccaPlanner import YuccaPlanner


class YuccaPlanner_MaxSize(YuccaPlanner):
    def __init__(
        self,
        task,
        preprocessor="YuccaPreprocessor",
        threads=None,
        disable_cc_analysis=True,
        disable_sanity_checks=False,
        view=None,
    ):
        super().__init__(task, preprocessor, threads, disable_cc_analysis, disable_sanity_checks, view)
        self.name = str(self.__class__.__name__) + str(view or "")

    def determine_target_size_from_fixed_size_or_spacing(self):
        self.fixed_target_size = self.dataset_properties["original_max_size"]
        self.fixed_target_spacing = None
