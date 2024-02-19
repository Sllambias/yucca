from yucca.planning.YuccaPlanner import YuccaPlanner


class YuccaPlanner_MaxSize(YuccaPlanner):
    def __init__(
        self,
        task,
        preprocessor="YuccaPreprocessor",
        threads=2,
        disable_sanity_checks=False,
        disable_unittests=False,
        view=None,
    ):
        super().__init__(task, preprocessor, threads, disable_sanity_checks, disable_unittests, view)
        self.name = str(self.__class__.__name__) + str(view or "")

    def determine_target_size_from_fixed_size_or_spacing(self):
        self.fixed_target_size = self.dataset_properties["original_max_size"]
        self.fixed_target_spacing = None
