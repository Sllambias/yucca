from yucca.planning.YuccaPlanner import YuccaPlanner


class YuccaPlanner_MissingModalities(YuccaPlanner):
    def __init__(self, task, preprocessor=None, threads=None, disable_sanity_checks=False, view=None, allow_missing_modalities=None):
        super().__init__(
            task, preprocessor=preprocessor, threads=threads, disable_sanity_checks=disable_sanity_checks, view=view, allow_missing_modalities=allow_missing_modalities,
        )
        self.name = str(self.__class__.__name__)
        self.allow_missing_modalities = True
