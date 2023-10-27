<div align="center">

<img src="https://github.com/Sllambias/yucca/assets/9844416/dc37d3c0-5181-4bb2-9630-dee9bc67165e" width="368" height="402" />

[![Paper](https://img.shields.io/badge/arXiv-1904.10620-44cc11.svg)](https://arxiv.org/abs/1904.10620)
![coverage](https://img.shields.io/badge/coverage-80%25-yellowgreen)
![downloads](https://img.shields.io/badge/downloads-13k%2Fmonth-brightgreen)
![version](https://img.shields.io/badge/version-1.2.3-blue)

</div>

# Yucca
All-Purpose Segmentation Framework. Yucca is designed to be plug-and-play while still being heavily customizable. This allows users to employ the basic Yucca models as solid baselines but it also allows users to change and experiment with exact features in a robust and thoroughly tested research environment. The Yucca project is inspired by Fabien Isensee's [nnUNet](https://github.com/MIC-DKFZ/nnUNet).

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

Create a python=3.10 environment for Yucca to avoid conflicts with other projects. 

IMPORTANT: First install Pytorch for GPU following appropriate instructions from e.g. https://pytorch.org/get-started/locally/.
Then navigate to Yucca and install the package from there.

For an Ubuntu system with Cuda 11.7:
```
> conda create -n yuccaenv python=3.10
> conda activate yuccaenv
> conda install -c anaconda setuptools
> conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
> conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
> cd Yucca
> pip install -e .
```
To use other CUDA or PyTorch versions refer to 1. for the current PyTorch installation, 2. for previous versions and 3. for the appropriate CUDA toolkit. Note that the CUDA versions used in the PyTorch and CUDA-toolkit installations should match (in the example above both use 11.7) 

1. https://pytorch.org/get-started/locally/
2. https://pytorch.org/get-started/previous-versions/
3. https://anaconda.org/nvidia/cuda-toolkit

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

Prior to preprocessing and training all datasets must be converted to Yucca-compliant tasks. This is done to ensure reproducibility and eliminate data leakage. For a tutorial see the [Task Conversion Guide](yucca/documentation/tutorials/task_conversion.md).

# Preprocessing

Preprocessing is carried out using the *yucca_preprocess* command, which calls the [run_preprocessing](yucca/run/run_preprocessing.py) script. For help and all the available arguments see the output of the -h flag below.

```console
> yucca_preprocess -h
usage: yucca_preprocess [-h] -t TASK [-pl PL] [--disable_unit_tests DISABLE_UNIT_TESTS] [--threads THREADS]

options:
  -h, --help            show this help message and exit
  -t TASK, --task TASK  Name of the task to preprocess. Should be of format: TaskXXX_MYTASK
  -pl PL                Experiment Planner Class to employ. Defaults to the YuccaPlanner
  --disable_unit_tests DISABLE_UNIT_TESTS
                        Enable or disable unittesting
```

An example of preprocessing a task called Task001_Brains with the default planner:
```
> yucca_preprocess -t Task001_Brains
```

Internally, the *yucca_preprocess* command calls the Planner and Preprocessor classes.
Initially, the appropriate Planner is called. This is by default the [YuccaPlannerV2](yucca/planning/YuccaPlannerV2.py). The Planner defines the normalization operation, spacing/resolution and orientation and saves relevant properties in a .pkl file for later use.

Afterwards, the Preprocessor is called. This is by default the [YuccaPreprocessor](yucca/preprocessing/YuccaPreprocessor.py). This preprocesses training data according to the operations and values supplied by the chosen planner. As such the Preprocessor should very rarely be changed, while the Planner will often be changed to employ alternative preprocessing schemes.

# Training

Training is carried out using the *yucca_train* command, which calls the [run_training](yucca/run/run_training.py) script. Prior to training the *yucca_preprocessing* command must be used to preprocess data and create the appropriate plan folder and *_plans.json* file. For help and all the available arguments see the output of the -h flag below.

```console
> yucca_train -h
usage: yucca_train [-h] [-t TASK] [-m M] [-d D] [-tr TR] [-pl PL] [-f F] [--lr LR] [--loss LOSS] [--mom MOM] [--continue_train]

options:
  -h, --help            show this help message and exit
  -t TASK, --task TASK  Name of the task used for training. The data should already be preprocessed using yucca_preprocessArgument should be of format: TaskXXX_MYTASK
  -m M                  Model Architecture. Should be one of MultiResUNet or UNet Note that this is case sensitive. Defaults to the standard UNet.
  -d D                  Dimensionality of the Model. Can be 3D, 25D or 2D. Defaults to 3D.
  -tr TR                Trainer Class to be used. Defaults to the basic YuccaTrainer
  -pl PL                Plan ID to be used. This specifies which plan and preprocessed data to use for training on the given task. Defaults to the YuccaPlans folder
  -f F                  Fold to use for training. Unless manually assigned, folds [0,1,2,3,4] will be created automatically. Defaults to training on fold 0
  --lr LR               Should only be used to employ alternative Learning Rate. Format should be scientific notation e.g. 1e-4.
  --loss LOSS           Should only be used to employ alternative Loss Function
  --mom MOM             Should only be used to employ alternative Momentum.
  --continue_train      continue training a previously saved checkpoint.
```
To start training using the default Trainer Class and plans file simply supply (1) the task with -t (2) the model architecture with -m and (3) the dimensions with -d. An example of training on a task called Task001_Brains, using a 3D MultiResUnet:
```
> yucca_train -t Task001_Brains -m MultiResUNet
```

By default this will use the [YuccaTrainerV2](yucca/training/trainers/YuccaTrainerV2.py). To change Trainer use the -tr flag. The Trainer Class defines the model training parameters. This includes: learning rate, learning rate scheduler, momentum, loss function, optimizer, batch size, epochs, foreground oversampling, patch size and data augmentation scheme. To change the values of these parameters see [Changing Parameters](yucca/documentation/tutorials/changing_parameters.md#model--training). To use plan files created by non-default Planners use -pl and specify the name of the alternative plan file. To specify which fold to train on use -f. By default training is done on fold 0 (unless manually specified we create 5 random splits).

An example of training on a task called Task002_NotBrains, using a 2D MultiResUnet with the CustomTrainer and CustomPlans on fold 3:
```
> yucca_train -t Task002_NotBrains -m MultiResUNet -d 2D -tr CustomTrainer -pl CustomPlans -f 3
```

# Inference

Inference is carried out using the *yucca_inference* command, which calls the [run_inference](yucca/run/run_inference.py) script. This relies on models previously trained using the *yucca_train* command. For help and all the available arguments see the output of the -h flag below. 

```console
> yucca_inference -h
usage: yucca_inference [-h] -s S -t T [-f F] [-m M] [-d D] [-tr TR] [-pl PL] [-chk CHK] [--ensemble] [--not_strict] [--save_softmax] [--overwrite] [--no_eval] [--predict_train]

options:
  -h, --help       show this help message and exit
  -s S             Name of the source task i.e. what the model is trained on. Should be of format: TaskXXX_MYTASK
  -t T             Name of the target task i.e. the data to be predicted. Should be of format: TaskXXX_MYTASK
  -f F             Select the fold that was used to train the model desired for inference. Defaults to looking for a model trained on fold 0.
  -m M             Model Architecture. Defaults to UNet.
  -d D             2D, 25D or 3D model. Defaults to 3D.
  -tr TR           Full name of Trainer Class. e.g. 'YuccaTrainer_DCE' or 'YuccaTrainerV2'. Defaults to YuccaTrainerV2.
  -pl PL           Plan ID. Defaults to YuccaPlannerV2
  -chk CHK         Checkpoint to use for inference. Defaults to checkpoint_best.
  --ensemble       Used to initialize data preprocessing for ensemble/2.5D training
  --not_strict     Strict determines if all expected modalities must be present, with the appropriate suffixes (e.g. '_000.nii.gz'). Only touch if you know what you're doing.
  --save_softmax   Save softmax outputs. Required for softmax fusion.
  --overwrite      Overwrite existing predictions
  --no_eval        Disable evaluation and creation of metrics file (result.json)
  --predict_train  Predict on the training set. Useful for debugging.
```

To run inference using the default Trainer Class, plan file and folds supply (1) the source task (what the model is trained on) with -s (2) the target task (what we want to predict) with -t (3) the model architecture with -m and (4) the dimensions with -d.

An example of running inference on the test set of a task called Task001_Brains, using a 3D MultiResUnet trained on the same task:
```
> yucca_inference -s Task001_Brains -t Task001_Brains -m MultiResUNet
```
If, on the other hand, we were to run inference on a new task called Task002_NotBrains, using a 2D UNet trained on Task001_Brains with a custom Trainer Class:
```
> yucca_inference -s Task001_Brains -t Task002_NotBrains -d 2D -tr CustomTrainer
```

# Ensembles

Using an ensemble of models can be achieved manually and automatically. 
The automatic method trains 3 models on the axial, sagittal and coronal views and combine these in inference. It is called using the "--ensemble" flag available for [yucca_preprocess](#preprocessing), [yucca_train](#training) and [yucca_inference](#inference). The preprocessor will preprocess the selected Task 3 times, one for each of the views, and save the preprocessed datasets in separate folders with the suffixes "X", "Y" and "Z", e.g. 'yucca_preprocessed/TaskXXX_MYTASK/YuccaPlannerV2X'. The trainer will train a model on each of the three views and save the trained models using the same suffixes, e.g. 'yucca_models/TaskXXX_MYTASK/YuccaTrainerV2_YuccaPlannerV2X'. Then, the softmax outputs from the three models are fused to create the final prediction and saved in a folder suffixed by "_Ensemble", e.g. 'yucca_results/TaskXXX_MYTASK/UNet2D/YuccaTrainerV2_YuccaPlannerV2_Ensemble'.

The manual method can be used to speed up the process or use different ensembles og e.g. different folds rather than views. The speedup comes from the fact that the --ensemble flag isn't capable of handling parallel model training, it is therefore quite slow as models will only be trained sequentially. To "parallelize" this, you can train models on each of the three preprocessed datasets:
```console
> yucca_train -t Task002_NotBrains -d 2D -pl YuccaPlannerV2X
> yucca_train -t Task002_NotBrains -d 2D -pl YuccaPlannerV2Y
> yucca_train -t Task002_NotBrains -d 2D -pl YuccaPlannerV2Z
```
Preprocessing and inference can also be "parallelized" in the same way, but that will rarely be necessary for speed optimization as these are fairly fast operations. 




