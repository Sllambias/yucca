<div align="center">

<img src="https://github.com/Sllambias/yucca/assets/9844416/dc37d3c0-5181-4bb2-9630-dee9bc67165e" width="368" height="402" />

</div>

# Yucca

Yucca is a modular machine learning framework built on PyTorch and PyTorch Lightning, presented in our paper [here](https://arxiv.org/abs/2407.19888), and inspired by Fabien Isensee's [nnUNet](https://github.com/MIC-DKFZ/nnUNet) and implemented for end-to-end medical imaging applications. This includes preprocessing volumetric data, training segmentation and self-supervised models, running inference and evaluation, and managing folder structure and naming conventions. 

Yucca supports (1) external projects importing individual Yucca components, (2) standalone Yucca-based projects e.g. using the preprocessing, training, and inference template scripts, or (3) projects employing the CLI-based end-to-end Yucca implementation, illustrated in the [diagram](#yucca). To cater to our different users Yucca features a three-tiered architecture: Functional, Modules, and Pipeline.

The Functional tier is inspired by torch.nn.functional and consists solely of stateless functions. This tier shapes the foundational building blocks of the framework, providing essential operations without maintaining any internal state. These functions are designed to be simple and reusable, allowing users to build custom implementations from scratch. The components are modular and can be easily tested and debugged by focusing on pure functions. 

The Modules tier is responsible for composing functions established in the Functional tier with logic, and conventions. Modules introduce a layer of structure, handling the organization and processing of inputs and outputs. They encapsulate specific functionalities and are designed to be more user-friendly, reducing the complexity involved in building custom models. While modules rely on more assumptions about the data, they still offer significant flexibility for customization and extension. 

The Pipeline tier represents our interpretation of an end-to-end implementation, built upon the previous two tiers. The Pipeline offers the end-to-end capabilities known from nnU-Net, while also allowing for effortless customization, as supported by the comprehensive documentation found in [Changing Parameters](yucca/documentation/guides/changing_parameters.md#model--training). 

Our Pipeline allows users to quickly train solid baselines or change features to conduct experiments on individual components in a robust and thoroughly tested research environment. For situations where full control is required, or simply desired, the Functional and Modules tiers are better suited. These tiers serve the advanced machine learning practitioners, wishing to import building blocks with which they can build their own house.

![alt text](yucca/documentation/illustrations/yucca_diagram.svg?raw=true)

# Table of Contents
- [Guides](#guides)
- [Installation](#installation)
- [Introduction to Yucca](#introduction-to-yucca)
- [Task Conversion](#task-conversion)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Inference](#inference)
- [Referencing Yucca](#referencing)

# Guides

- [Changing Pipeline Parameters](yucca/documentation/guides/changing_pipeline_parameters.md#model--training)
- [Classification](yucca/documentation/guides/classification.md)
- [Environment Variables](yucca/documentation/guides/environment_variables.md)
- [Ensembles](yucca/documentation/guides/ensembles.md)
- [FAQ](yucca/documentation/guides/FAQ.md)
- [Run Scripts Advanced](yucca/documentation/guides/run_scripts_advanced.md)
- [Task Conversion](yucca/documentation/guides/task_conversion.md)
- [Unsupervised](yucca/documentation/guides/unsupervised.md)

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

When W&B is enabled Yucca will automatically generate plots and illustrations and upload these to your personal Yucca project. This happens while your experiments are running, and you'll find pages that look somewhat similar to the example screenshot found [here](yucca/documentation/illustrations/WB_Example.pdf).

Setting up W&B is very simple.
First navigate to https://wandb.ai/home and log in or sign up for Weights and Biases.
Then activate the appropriate environment, install Weights and Biases and log in by following the instructions (i.e. paste the key from https://wandb.ai/authorize into the terminal).
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
  2. **-pl**: The Planner class, which is responsible for determining *what* we do in preprocessing and *how* it is done. This includes setting the normalization, resizing, resampling and transposition operations and any values associated with them. The planner class defaults to the `YuccaPlanner`, but it can also be any custom planner found or created in the [Planner directory](yucca/pipeline/planning) and its subdirectories.
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
  3. **-m**: The model architecture. This includes any model implemented in the [Model directory](yucca/networks/networks). Including, but not limited to, `U-Net`, `UNetR`, `MultiResUNet` and `ResNet50`.
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

## Referencing

When referring to this work please cite [this paper](https://arxiv.org/abs/2407.19888):

Llambias, S. N., Machnio, J., Munk, A., Ambsdorf, J., Nielsen, M., & Ghazi, M. M. (2024). Yucca: A deep learning framework for medical image analysis. arXiv preprint arXiv:2407.19888.

