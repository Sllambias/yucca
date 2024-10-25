## Classification models

Beware: The classification implementation is untrodden grounds and no guarantees on performance can therefore be made. 

Training classification models is carried out by:
  1. Converting your raw dataset to a Yucca compliant format with class labels in individual `.txt` files. See the [Task Conversion guide](yucca/documentation/guides/task_conversion.md) for instructions on how to convert your datasets.
  2. Selecting a Planner that:
    1. Always preprocesses the task converted dataset using the `ClassificationPreprocessor`, such as the [`ClassificationPlanner`](yucca/pipeline/planning/ClassificationPlanner.py). This preprocessor expects to find `.txt` files rather than image files in the label folders and it does not perform any preprocessing on the labels. Alternatively, the `ClassificationPreprocessor` can be selected using the `-pr ClassificationPreprocessor` flag in `yucca_preprocess` 
    2. Resamples images to a fixed target size, such as the [`YuccaPlanner_224x224`](yucca/pipeline/planning/resampling/YuccaPlanner_224x224.py). Having a fixed image size enables training models on full images, rather than patches of images. This is often necessary in classification where we want 1 (or very few) image-level prediction.
  3. Selecting a manager that trains models on full-size images. This is any manager with the ```patch_based_training=False```, such as the [`YuccaManager_NoPatches`](yucca/pipeline/managers/alternative_managers/YuccaManager_NoPatches.py).
  4. Selecting a model architecture that supports classification. Currently that is limited to the [`ResNet50`](yucca/networks/networks/resnet.py) but most networks can be adapted to support this with limited changes (in essence, this can be achieved by adding a Linear layer with input channels equal to the flattened output of the penultimate layer and output channels equals to the number of classes in the dataset).
  5. Running `yucca_inference` with the `--task_type classification` flag. 
