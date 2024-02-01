<div align="center">

<img src="https://github.com/Sllambias/yucca/assets/9844416/dc37d3c0-5181-4bb2-9630-dee9bc67165e" width="368" height="402" />

[![Paper](https://img.shields.io/badge/arXiv-1904.10620-44cc11.svg)](https://arxiv.org/abs/1904.10620)
![coverage](https://img.shields.io/badge/coverage-80%25-yellowgreen)
![downloads](https://img.shields.io/badge/downloads-13k%2Fmonth-brightgreen)
![version](https://img.shields.io/badge/version-1.2.3-blue)

</div>

# Yucca
End-to-end modular machine learning framework for classification, segmentation and unsupervised learning. Yucca is designed to be plug-and-play while still allowing for effortless customization. This allows users to employ the basic Yucca models as solid baselines, but it also allows users to change and experiment with exact features in a robust and thoroughly tested research environment. The Yucca project is inspired by Fabien Isensee's [nnUNet](https://github.com/MIC-DKFZ/nnUNet).

![alt text](yucca/documentation/illustrations/end_to_end_diagram.svg?raw=true)

# Table of Contents
- [Guides](#guides)
- [Installation](#installation)
- [Introduction to Yucca](#introduction-to-yucca)
- [Task Conversion](#task-conversion)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Inference](#inference)
- [Ensembles](#ensembles)
- [Classification](#classification-models)
- [Segmentation](#segmentation-models)
- [Unsupervised](#unsupervised-models)

# Guides

- [Changing Parameters](yucca/documentation/guides/changing_parameters.md#model--training)
- [Environment Variables](yucca/documentation/guides/environment_variables.md)
- [FAQ](yucca/documentation/guides/FAQ.md)
- [Run Scripts Advanced](yucca/documentation/guides/run_scripts_advanced.md)
- [Task Conversion](yucca/documentation/guides/task_conversion.md)
 
# Installation

## Install an editable version of the project with Cuda support using Conda

Create a python=3.10 or python=3.11 environment exclusively for Yucca to avoid conflicts with other projects. 

IMPORTANT: First install Pytorch for GPU following appropriate instructions from e.g. https://pytorch.org/get-started/locally/.
Then navigate to Yucca and install the package from there.

For an Ubuntu system with Cuda=>12.1 and python=3.11:
```
> git clone https://github.com/Sllambias/yucca.git
> conda create -n yuccaenv python=3.11
> conda activate yuccaenv
> conda install -c anaconda setuptools
> conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit
> conda install pytorch==2.1.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
> cd yucca
> pip install -e .
```

To use other CUDA or PyTorch versions refer to 1. for the current PyTorch installation, 2. for previous versions and 3. for the appropriate CUDA toolkit. Note that the CUDA versions used in the PyTorch and CUDA-toolkit installations should match (in the example above both use 12.1).

1. https://pytorch.org/get-started/locally/
2. https://pytorch.org/get-started/previous-versions/
3. https://anaconda.org/nvidia/cuda-toolkit

## Install the package as a dependency in another project
If you just want to install Yucca locally on your computer, use
```
pip install git+https://github.com/Sllambias/yucca.git
```
this will install the code from github, not an eventual local clone.


# Weights & Biases
Weights & Biases is the main tool for experiment tracking in Yucca. It is extremely useful to understand how your models are behaving and often also why. Although it can be disabled, it is heavily encouraged to install and use it with Yucca.

Navigate to https://wandb.ai/home and log in or sign up for Weights and Biases.
Activate the appropriate environment, install Weights and Biases and log in by following the instructions (i.e. paste the key from https://wandb.ai/authorize into the terminal).
```console
> conda activate yuccaenv
> pip install wandb
> wandb login
wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
wandb: You can find your API key in your browser here: https://wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit:
```

# Introduction to Yucca

The Yucca pipeline comprises the 4 processes illustrated in the [diagram](#yucca). In the first step, the user is expected to prepare the data for Yucca. In the remaining three steps, Yucca will take over regarding file management.
  1. **The Task Conversion** step requires that the user _converts_ their arbitrarily structured data to the file and folder structure Yucca requires. From now on, Yucca will handle the data. Task Conversion involves moving and renaming the data along with creating a metadata file.
  2. **The Preprocessing step** takes the Task Converted data and preprocesses it and then subsequently saves it in its preprocessed state in the format expected by the Yucca training process.
  3. **The Training step** takes the preprocessed data and trains a model, and then subsequently saves it along with its checkpoints and metadata.
  4. **The Inference step** takes the trained model and applies it to a task-converted (but not preprocessed) test set. During inference, the unseen samples are preprocessed with the same preprocessor used in the preprocessing step. Predictions are then saved. When inference is concluded, the predictions are evaluated against the ground truth, and a .json file containing the results is saved next to the predictions.

## Environment Variables

Initially, the environment variables used in Yucca must be defined. To set these, see the [Environment Variables](yucca/documentation/guides/environment_variables.md) guide. 

## Task Conversion

Before preprocessing and training, all datasets must be converted to Yucca-compliant tasks. This is done to ensure reproducibility and eliminate data leakage. For a tutorial see the [Task Conversion Guide](yucca/documentation/guides/task_conversion.md).

## Preprocessing

Preprocessing is carried out using the `yucca_preprocess` command. For advanced usage see: [`run_scripts_advanced.py`](yucca/documentation/guides/run_scripts_advanced.md#preprocessing)

Basic Yucca preprocessing relies on three CLI flags:
  1. **-t**: The target task-converted raw data to be preprocessed.
  2. **-pl**: The Planner class, which is responsible for determining *what* we do in preprocessing and *how* it is done. This includes setting the normalization, resizing, resampling and transposition operations and any values associated with them. The planner class defaults to the `YuccaPlanner`, but it can also be any custom planner found or created in the [Planner directory](yucca/planning) and its subdirectories.
  3. **-pr**: The Preprocessor class. The Preprocessor is a workhorse that receives an instruction manual from the Planner, which it carries out. The Preprocessor can be one of `YuccaPreprocessor` (default), `ClassificationPreprocessor` and `UnsupervisedPreprocessor`. The only aspect in which they differ is how they expect the ground truth to look. The `YuccaPreprocessor` expects to find images, the `ClassificationPreprocessor` expects to find .txt files with image-level classes and the `UnsupervisedPreprocessor` expects not to find any ground truth. 

An example of preprocessing a task called `Task001_Brains` with the default planner and the `ClassificationPreprocessor`:
```
> yucca_preprocess -t Task001_Brains -pr ClassificationPreprocessor
```

## Training

Training is carried out using the `yucca_train` command. For advanced usage see: [`run_scripts_advanced.py`](yucca/documentation/guides/run_scripts_advanced.md#training). Before training any models, a preprocessed dataset must be prepared using the `yucca_preprocessing` command.

Basic Yucca training relies on five CLI flags:
  1. **-t**: The target preprocessed data on which the model will be trained.
  2. **-d**: The model dimensions. This can be either 2D or 3D (default).
  3. **-m**: The model architecture. This includes any model implemented in the [Model directory](yucca/network_architectures/networks). Including, but not limited to, `U-Net`, `UNetR`, `MultiResUNet` and `ResNet50`.
  4. **-man**: The Manager to use. This defaults to the `YuccaManager`.
  5. **-pl**: The Planner used to preprocess the training data. This defaults to the `YuccaPlanner`.

An example of training a `MultiResUNet` with the default Manager on a task called `Task001_Brains` that has been preprocessed using the default `YuccaPlanner`:
 using a 2D `MultiResUnet`:
```
> yucca_train -t Task001_Brains -m MultiResUNet -d 2D
```

## Inference

Inference is carried out using the `yucca_inference` command. For advanced usage see: [`run_scripts_advanced.py`](yucca/documentation/guides/run_scripts_advanced.md#inference). Prior to inference, the model must be trained using the `yucca_train` command, and the target dataset must be task-converted.

Basic Yucca inference relies on six CLI flags.
  1. **-t**: The target task-converted raw data on which to run inference.
  2. **-s**: The source task on which the model was trained.
  3. **-d**: The dimensions of the trained model.
  4. **-m**: The architecture of the trained model.
  5. **-man**: The Manager to use. This defaults to the `YuccaManager`.
  6. **-pl**: The Planner used to preprocess the training data.

An example of running inference on the test set of a task called `Task001_Brains`, using a 3D `MultiResUnet` trained on the train set of the same task:
```
> yucca_inference -t Task001_Brains -s Task001_Brains -m MultiResUNet
```
An example of running inference on the test set of a task called `Task002_Lungs`, using a 2D `UNet` trained on a task called `Task001_Brains`:
```
> yucca_inference -t Task002_NotBrains -s Task001_Brains -d 2D -m UNet
```

## Ensembles

To train an ensemble of models we use the `yucca_preprocess`, `yucca_train` and `yucca_inference` commands. For advanced usage see: [`run_scripts_advanced.py`](yucca/documentation/guides/run_scripts_advanced.md#ensembles). A common application of model ensembles is to train 2D models on each of the three axes of 3D data (either denoted as the X-, Y- and Z-axis or, in medical imaging, the axial, sagittal and coronal views) and then fuse their predictions in inference. 

To train 3 models on the three axes of a 3D dataset called `Task001_Brains` prepare three preprocessed versions of the dataset using the three Planners `YuccaPlannerX`, `YuccaPlannerY` and `YuccaPlannerZ`:
```console
> yucca_preprocess -t Task001_Brains -pl YuccaPlannerX
> yucca_preprocess -t Task001_Brains -pl YuccaPlannerY
> yucca_preprocess -t Task001_Brains -pl YuccaPlannerZ
```

Then, train three 2D models one on each version of the preprocessed dataset:
```console
> yucca_train -t Task001_Brains -pl YuccaPlannerX -d 2D
> yucca_train -t Task001_Brains -pl YuccaPlannerY -d 2D
> yucca_train -t Task001_Brains -pl YuccaPlannerZ -d 2D
```

Then, run inference on the target dataset with each trained model.
```console
> yucca_inference -t Task001_Brains -pl YuccaPlannerX -d 2D
> yucca_inference -t Task001_Brains -pl YuccaPlannerY -d 2D
> yucca_inference -t Task001_Brains -pl YuccaPlannerZ -d 2D
```

Finally, fuse their results and evaluate the predictions.
```console
> yucca_ensemble --in_dirs /path/to/predictionsX /path/to/predictionsY /path/to/predictionsZ --out_dir /path/to/ensemble_predictionsXYZ
```

## Classification models

Training classification models is carried out by:
  1. Converting your raw dataset to a Yucca compliant format with class labels in individual `.txt` files. See the [Task Conversion guide](yucca/documentation/guides/task_conversion.md) for instructions on how to convert your datasets.
  2. Selecting a Planner that:
    1. Always preprocesses the task converted dataset using the `ClassificationPreprocessor`, such as the [`ClassificationPlanner`](yucca/planning/ClassificationPlanner.py). This preprocessor expects to find `.txt` files rather than image files in the label folders and it does not perform any preprocessing on the labels. Alternatively, the `ClassificationPreprocessor` can be selected using the `-pr ClassificationPreprocessor` flag in `yucca_preprocess` 
    2. Resamples images to a fixed target size, such as the [`YuccaPlanner_224x224`](yucca/planning/resampling/YuccaPlanner_224x224.py). Having a fixed image size enables training models on full images, rather than patches of images. This is often necessary in classification where we want 1 (or very few) image-level prediction.
  3. Selecting a manager that trains models on full-size images. This is any manager with the ```patch_based_training=False```, such as the [`YuccaManager_NoPatches`](yucca/training/managers/alternative_managers/YuccaManager_NoPatches.py).
  4. Selecting a model architecture that supports classification. Currently that is limited to the [`ResNet50`](yucca/network_architectures/networks/resnet.py) but most networks can be adapted to support this with limited changes (in essence, this can be achieved by adding a Linear layer with input channels equal to the flattened output of the penultimate layer and output channels equals to the number of classes in the dataset).
  5. Running `yucca_inference` with the `--task_type classification` flag. 

## Segmentation models

Training segmentation models is carried out by following the standard procedure introduced in the [Introduction to Yucca](yucca)

## Unsupervised models

Training Unsupervised models is carried out by:
  1. Converting your raw dataset to a Yucca compliant format with no label files. See the [Task Conversion guide](yucca/documentation/guides/task_conversion.md) for instructions on how to convert your datasets.
  2. Selecting a Planner that always preprocesses the task converted dataset using the `UnsupervisedPreprocessor`, such as the [`UnsupervisedPlanner`](yucca/planning/YuccaPlanner.py). This preprocessor expects to find no label files. Alternatively, the `UnsupervisedPreprocessor` can be selected using the `-pr UnsupervisedPreprocessor` flag in `yucca_preprocess`.

When models are trained on a dataset preprocessed with the UnsupervisedPreprocessor, Yucca will use the `unsupervised` preset in the [`YuccaAugmentationComposer`](yucca/training/augmentation/YuccaAugmentationComposer.py). This sets (1) `skip_label` to True (which means we don't expect a label in the array), (2) `copy_image_to_label` to True, which means the image data is copied to also be the label data (the image is copied after applying normal augmentations) and finally, (3) `mask_image_for_reconstruction` to True, which means we randomly mask the image data (this is applied AFTER the image is copied to the label). 
