# Compression
Yucca normally stores training data uncompressed and everything else compressed. Both raw data and predictions are stored in .nii.gz, .npz, .jpg, .png or similar file formats files while the preprocessed data is kept as .npy files (rather than the compressed .npz counterpart). During model training the preprocessed data will be read many, many times and using compressed data for this will significantly slow down training steps.

If it is not possible to store the training data uncompressed due to storage limitations it is possible to train on compressed data.

To achieve this use a Planner with compression enabled, such as the YuccaPlanner_Compress, or create one yourself with the desired Planner parameters.
```
from yucca.pipeline.planning.YuccaPlanner import YuccaPlanner


class YuccaPlanner_Compress(YuccaPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = str(self.__class__.__name__)
        self.compress = True
```

