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
- [Task Conversion](#task-conversion)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Inference](#inference)
- [Ensembles](#ensembles)

# Guides

- [Task Conversion](yucca/documentation/tutorials/task_conversion.md)
- [Changing Parameters](yucca/documentation/tutorials/changing_parameters.md#model--training)
- [Environment Variables](yucca/documentation/tutorials/environment_variables.md)

# Installation

## Install an editable version of the project with Cuda support using Conda

Create a python=3.10 environment exclusively for Yucca to avoid conflicts with other projects. 

IMPORTANT: First install Pytorch for GPU following appropriate instructions from e.g. https://pytorch.org/get-started/locally/.
Then navigate to Yucca and install the package from there.

For an Ubuntu system with Cuda=>12.1:
```
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

# Usage

The Yucca pipeline is comprised of the 4 processes illustrated in the [diagram](#yucca). In the first step the user is expected to prepare the data for Yucca. In the remaining three steps Yucca will take over with regards to file management.
  1. The Task Conversion step requires that the user _converts_ their arbitrarly structured data to the file and folder structure required by Yucca. Hereafter, Yucca handles the data. Task Conversion involves moving and renaming the data along with creating a metadata file.
  2. The Preprocessing step takes the Task Converted data and preprocesses it and then subsequently saves it in its preprocessed state in the format expected by the Yucca training process.
  3. The Training step takes the preprocessed data and trains a model and then subsequently saves it along with its checkpoints and metadata.
  4. The Inference step takes the trained model and applies it to a task converted (but not preprocessed) test set. During inference the unseen samples are preprocessed with the same preprocessor used in the preprocessing step. Predictions are then saved. When inference is concluded the predictions are evaluated against the ground truth and a .json file containing the results are saved next to the predictions.

# Environment Variables

Initially, the environment variables used in Yucca must be defined. To set these see: [Environment Variables](yucca/documentation/tutorials/environment_variables.md). 

# Task Conversion

Prior to preprocessing and training all datasets must be converted to Yucca-compliant tasks. This is done to ensure reproducibility and eliminate data leakage. For a tutorial see the [Task Conversion Guide](yucca/documentation/tutorials/task_conversion.md).

# Preprocessing

Preprocessing is carried out using the `yucca_preprocess` command. For advanced usage see: [`run_scripts_advanced.py`](yucca/documentation/tutorials/run_scripts_advanced.md#preprocessing)

Basic Yucca preprocessing relies on three components. 
  1. The target task-converted raw data to be preprocessed specified with the required **-t** flag.
  2. The Planner class. The Planner is responsible for determining *what* we do in preprocessing and *how* it is done. This includes setting the normalization, resizing, resampling and transposition operations and any values associated with them. The planner class defaults to the `YuccaPlanner` but it can also be any custom planner found or created in the [Planner directory](yucca/planning) and its subdirectories.  The planner can be changed using the **-pl** flag.  
  3. The Preprocessor class. The Preprocessor is a work horse, that receives an instruction manual from the Planner which it carries out. The Preprocessor can be one of `YuccaPreprocessor` (default), `ClassificationPreprocessor` and `UnsupervisedPreprocessor`. The only aspect in which they differ is how they expect the ground truth to look. The `YuccaPreprocessor` expects to find images, the `ClassificationPreprocessor` expects to find .txt files with image-level classes and the `UnsupervisedPreprocessor` expects to not find any ground truth. This can be changed using the **-pr** flag. 

An example of preprocessing a task called `Task001_Brains` with the default planner and the `ClassificationPreprocessor`:
```
> yucca_preprocess -t Task001_Brains -pr ClassificationPreprocessor
```

# Training

Training is carried out using the `yucca_train` command. For advanced usage see: [`run_scripts_advanced.py`](yucca/documentation/tutorials/run_scripts_advanced.md#training). Prior to training any models a preprocessed dataset must be prepared using the `yucca_preprocessing` command.

Basic Yucca training relies on four components.
  1. The target preprocessed data on which the model will be trained. This is specified using the **-t** flag.
  2. The model architecture. This includes any model implemented in the [Model directory](yucca/network_architectures/networks). Including, but not limited to, `U-Net`, `UNetR`, `MultiResUNet` and `ResNet50`. Specified by the **-m** flag.
  3. The model dimensions. This can be either 2D or 3D (default) and is specified with the **-d** flag.
  4. The planner used to determine preprocessing. This defaults to the `YuccaPlanner` but can be any planner found or created in the [Planner directory](yucca/planning).

An example of training a `MultiResUNet` on a task called `Task001_Brains` that has been preprocessed using the default `YuccaPlanner`:
 using a 2D `MultiResUnet`:
```
> yucca_train -t Task001_Brains -m MultiResUNet -d 2D
```


# Inference

Inference is carried out using the `yucca_inference` command. For advanced usage see: [`run_scripts_advanced.py`](yucca/documentation/tutorials/run_scripts_advanced.md#inference). Prior to inference the model must be trained using the `yucca_train` command and the target dataset must be task-converted.

Basic Yucca inference relies on four components.
  1. The target task-converted raw data on which to run inference. This is specified using the **-t** flag.
  2. The source task on which the model was trained. This is specified using the **-s** flag.
  3. The architecture of the trained model. Specified by the **-m** flag.
  4. The dimensions of the trained model. Specified by the **-d** flag.


To run inference using the default Trainer Class, plan file and folds supply (1) the source task (what the model is trained on) with `-s` (2) the target task (what we want to predict) with `-t` (3) the model architecture with `-m` and (4) the dimensions with `-d`.

An example of running inference on the test set of a task called `Task001_Brains`, using a 3D `MultiResUnet` trained on the train set of the same task:
```
> yucca_inference -t Task001_Brains -s Task001_Brains -m MultiResUNet
```
An example of running inference on the test set of a task called `Task002_Lungs`, using a 2D `UNet` trained on a task called `Task001_Brains`:
```
> yucca_inference -t Task002_NotBrains -s Task001_Brains -d 2D -m UNet
```

# Ensembles

Ensembling models is carried out using the `yucca_preprocess`, `yucca_train` and `yucca_inference` commands. For advanced usage see: [`run_scripts_advanced.py`](yucca/documentation/tutorials/run_scripts_advanced.md#ensembles). A common application of model ensembles is to train 2D models on each of the three axes of 3D data (either denoted as the X-, Y- and Z-axis or, in medical imaging, the axial, sagittal and coronal views) and then fuse their predictions in inference. 

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

# Classification

Loading...

# Unsupervised Training

Loading...
