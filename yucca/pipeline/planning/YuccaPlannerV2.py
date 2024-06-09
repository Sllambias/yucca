from yucca.pipeline.planning.YuccaPlanner import YuccaPlanner


class YuccaPlannerV2(YuccaPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = str(self.__class__.__name__)
        self.get_foreground_locations_per_label = True
