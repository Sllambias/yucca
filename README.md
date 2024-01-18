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

For an Ubuntu system with Cuda 11.7:
```
> conda create -n yuccaenv python=3.10
> conda activate yuccaenv
> conda install -c anaconda setuptools
> conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
> conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
> git clone https://github.com/Sllambias/yucca.git
> cd yucca
> pip install -e .
```
To use other CUDA or PyTorch versions refer to 1. for the current PyTorch installation, 2. for previous versions and 3. for the appropriate CUDA toolkit. Note that the CUDA versions used in the PyTorch and CUDA-toolkit installations should match (in the example above both use 11.7) 

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

# Task Conversion

Prior to preprocessing and training all datasets must be converted to a Yucca-compliant task. This is done to ensure reproducibility and eliminate data leakage. For a tutorial see the [Task Conversion Guide](yucca/documentation/tutorials/task_conversion.md).

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
  1. The source task on which the model was trained. This is specified using the **-s** flag.
  2. The target task-converted raw data on which to run inference. This is specified using the **-t** flag.
  3. The architecture of the trained model. Specified by the **-m** flag.
  4. The dimensions of the trained model. Specified by the **-d** flag.


To run inference using the default Trainer Class, plan file and folds supply (1) the source task (what the model is trained on) with `-s` (2) the target task (what we want to predict) with `-t` (3) the model architecture with `-m` and (4) the dimensions with `-d`.

An example of running inference on the test set of a task called `Task001_Brains`, using a 3D `MultiResUnet` trained on the train set of the same task:
```
> yucca_inference -s Task001_Brains -t Task001_Brains -m MultiResUNet
```
An example of running inference on the test set of a task called `Task002_Lungs`, using a 2D `UNet` trained on a task called `Task001_Brains`:
```
> yucca_inference -s Task001_Brains -t Task002_NotBrains -d 2D -m UNet
```

# Ensembles

Using an ensemble of models can be achieved manually and automatically. 
The automatic method trains 3 models on the axial, sagittal and coronal views and combine these in inference. It is called using the `--ensemble` flag available for [`yucca_preprocess`](#preprocessing), [yucca_train](#training) and [yucca_inference](#inference). The preprocessor will preprocess the selected Task 3 times, one for each of the views, and save the preprocessed datasets in separate folders with the suffixes `X`, `Y` and `Z`, e.g. `yucca_preprocessed/TaskXXX_MYTASK/YuccaPlannerV2X`. The trainer will train a model on each of the three views and save the trained models using the same suffixes, e.g. `yucca_models/TaskXXX_MYTASK/YuccaTrainerV2_YuccaPlannerV2X`. Then, the softmax outputs from the three models are fused to create the final prediction and saved in a folder suffixed by `_Ensemble`, e.g. `yucca_results/TaskXXX_MYTASK/UNet2D/YuccaTrainerV2_YuccaPlannerV2_Ensemble`.

The manual method can be used to speed up the process or use different ensembles og e.g. different folds rather than views. The speedup comes from the fact that the `--ensemble` flag isn't capable of handling parallel model training, it is therefore quite slow as models will only be trained sequentially. To "parallelize" this, you can train models on each of the three preprocessed datasets:
```console
> yucca_train -t Task002_NotBrains -d 2D -pl YuccaPlannerV2X
> yucca_train -t Task002_NotBrains -d 2D -pl YuccaPlannerV2Y
> yucca_train -t Task002_NotBrains -d 2D -pl YuccaPlannerV2Z
```
Preprocessing and inference can also be "parallelized" in the same way, but that will rarely be necessary for speed optimization as these are fairly fast operations. 




