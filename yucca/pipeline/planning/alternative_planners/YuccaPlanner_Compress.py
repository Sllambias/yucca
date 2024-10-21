from yucca.pipeline.planning.YuccaPlanner import YuccaPlanner


class YuccaPlanner_Compress(YuccaPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = str(self.__class__.__name__)
        self.compress = True
