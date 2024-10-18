## Unsupervised models

Training unsupervised models is carried out by:
  1. Converting your raw dataset to a Yucca compliant format with no label files. See the [Task Conversion guide](yucca/documentation/guides/task_conversion.md) for instructions on how to convert your datasets.
  2. Selecting a Planner that preprocesses the task converted dataset with the `UnsupervisedPreprocessor`, such as the [`UnsupervisedPlanner`](yucca/pipeline/planning/YuccaPlanner.py). This preprocessor expects to find no label files. Alternatively, the `UnsupervisedPreprocessor` can be selected using the `-pr UnsupervisedPreprocessor` flag in `yucca_preprocess`.

When models are trained on a dataset preprocessed with the UnsupervisedPreprocessor, Yucca will use the `unsupervised` preset in the [`YuccaAugmentationComposer`](yucca/data/augmentation/YuccaAugmentationComposer.py). This sets (1) `skip_label` to True (which means we don't expect a label in the array), (2) `copy_image_to_label` to True, which means the image data is copied to also be the label data (the image is copied after applying normal augmentations) and finally, (3) `mask_image_for_reconstruction` to True, which means we randomly mask the image data. This is applied AFTER the image is copied to the label. 
