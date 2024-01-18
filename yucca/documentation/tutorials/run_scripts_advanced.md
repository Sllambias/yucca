# Preprocessing

, which calls the [`run_preprocessing.py`](yucca/run/run_preprocessing.py) script

For help and all the available arguments see the output of the `-h` flag below.

```console
usage: yucca_preprocess [-h] -t TASK [-pl PL] [-pr PR] [-v V] [--ensemble] [--disable_sanity_checks DISABLE_SANITY_CHECKS]

options:
  -h, --help            show this help message and exit
  -t TASK, --task TASK  Name of the task to preprocess. Should be of format: TaskXXX_MYTASK
  -pl PL                Experiment Planner Class to employ. Defaults to the YuccaPlanner
  -pr PR                Preprocessor Class to employ. Defaults to the YuccaPreprocessor, but can be YuccaPreprocessor_CLS for classification tasks
  -v V                  Designate target view or orientation to obtain with transposition. Standard settings will handle this for you, but use this to manually specify. Can be 'X', 'Y' or 'Z'
  --ensemble            Used to initialize data preprocessing for ensemble/2.5D training
  --disable_sanity_checks DISABLE_SANITY_CHECKS
                        Enable or disable sanity checks
```


Internally, the `yucca_preprocess` command calls a planner and preprocessor class.
Initially, the appropriate planner is called. This is by default the [`YuccaPlanner`](yucca/planning/YuccaPlanner.py). The planner defines the normalization operation, spacing/resolution and orientation and saves relevant properties in a .pkl file for later use.

Afterwards, the preprocessor is called. This is by default the [`YuccaPreprocessor`](yucca/preprocessing/YuccaPreprocessor.py). This preprocesses training data according to the operations and values supplied by the chosen planner. As such the preprocessor should very rarely be changed, while the planner will often be changed to employ alternative preprocessing schemes.

# Training

For help and all the available arguments see the output of the `-h` flag below.

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


By default this will use the [YuccaTrainerV2](yucca/training/trainers/YuccaTrainerV2.py). To change Trainer use the `-tr` flag. The Trainer Class defines the model training parameters. This includes: learning rate, learning rate scheduler, momentum, loss function, optimizer, batch size, epochs, foreground oversampling, patch size and data augmentation scheme. To change the values of these parameters see [Changing Parameters](yucca/documentation/tutorials/changing_parameters.md#model--training). To use plan files created by non-default Planners use `-pl` and specify the name of the alternative plan file. To specify which fold to train on use `-f`. By default training is done on fold 0 (unless manually specified we create 5 random splits).

An example of training on a task called `Task002_NotBrains`, using a 2D `MultiResUnet` with the `CustomTrainer` and `CustomPlans` on fold 3:
```
> yucca_train -t Task002_NotBrains -m MultiResUNet -d 2D -tr CustomTrainer -pl CustomPlans -f 3
```

# Inference

For help and all the available arguments see the output of the `-h` flag below. 

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
