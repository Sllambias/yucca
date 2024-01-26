# Task Conversion
The `yucca_convert_task` supplies an API to access previously created Task Conversion scripts wrapped in `convert` functions.
Generally new users will need to write their own Task Conversion scripts but in some cases they might have already been written (e.g. for very popular Challenge datasets)

usage: yucca_convert_task [-h] -t TASK [-p PATH] [-d SUBDIR]

options:
  -h, --help            show this help message and exit
  -t TASK, --task TASK  Name of the task to preprocess. Should be of format: TaskXXX_MYTASK
  -p PATH, --path PATH  Path to source data
  -d SUBDIR, --subdir SUBDIR
                        Directory of data inside source data

The -t flag must point to the name of a previously created task conversion script, such as the [OASIS script](/yucca/yucca/task_conversion/Task001_OASIS.py). The -p flag points to the parent directory of the dataset folder. If left empty this defaults to [`yucca_source`](/yucca/yucca/documentation/tutorials/environment_variables.md). The -d flag points to the specific dataset directory inside the parent directory. 

If the OASIS dataset was located in /path/to/all/my/datasets/specific_dataset123 the command should be:
`yucca_convert_task -t Task001_OASIS -p /path/to/all/my/datasets -d specific_dataset123`

# Preprocessing

The `yucca_preprocess` CLI invokes the [`run_preprocessing.py`](yucca/run/run_preprocessing.py) script.
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

**-t**, **-pl** and **-pr** are covered in the front page [ReadMe](/yucca/README.md#preprocessing).

**-v**: Used to transpose images to a different orientation. Primarily used to train 2D models on a specific orientation.
**--ensemble**: Used to automatically preprocess 3 versions of the dataset using the X, Y and Z views. Can be manually obtained by preprocessing the dataset thrice with `-v X`, `-v Y` and `-v Z`.
**--disable_sanity_checks**: should only be used if you are aware something will violate sanity checks, but you want to still continue.

Internally, the `yucca_preprocess` command calls a planner and preprocessor class.
Initially, the appropriate planner is called. This is by default the [`YuccaPlanner`](yucca/yucca/planning/YuccaPlanner.py). The planner first collects dataset statistics can be used during preprocessing. These are saved in a `dataset_properties.pkl` file. Then, it specifies _what_ will happen during preprocessing. This includes the normalization operation, target spacing or size/resolution and orientation.

Afterwards, the preprocessor is called. This is by default the [`YuccaPreprocessor`](yucca/yucca/preprocessing/YuccaPreprocessor.py). This preprocesses training data according to the operations and values supplied by planner. As such the preprocessor should very rarely be changed, while the planner will often be changed to employ alternative preprocessing schemes.

# Training

The `yucca_train` CLI invokes the [`run_training.py`](yucca/run/run_training.py) script.
For help and all the available arguments see the output of the `-h` flag below.

```console
> yucca_train -h
usage: yucca_train [-h] [-t TASK] [-d D] [-m M] [-man MAN] [-pl PL] [--disable_logging] [--ds] [--epochs EPOCHS] [--experiment EXPERIMENT] [--loss LOSS] [--lr LR] [--mom MOM] [--new_version]
                   [--patch_size PATCH_SIZE] [--precision PRECISION] [--profile] [--split_idx SPLIT_IDX] [--split_data_method SPLIT_DATA_METHOD] [--split_data_param SPLIT_DATA_PARAM]
                   [--train_batches_per_step TRAIN_BATCHES_PER_STEP] [--val_batches_per_step VAL_BATCHES_PER_STEP]

options:
  -h, --help            show this help message and exit
  -t TASK, --task TASK  Name of the task used for training. The data should already be preprocessed using yucca_preprocessArgument should be of format: TaskXXX_MYTASK
  -d D                  Dimensionality of the Model. Can be 3D or 2D. Defaults to 3D. Note that this will always be 2D if ensemble is enabled.
  -m M                  Model Architecture. Should be one of MultiResUNet or UNet Note that this is case sensitive. Defaults to the standard UNet.
  -man MAN              Manager Class to be used. Defaults to the basic YuccaManager
  -pl PL                Plan ID to be used. This specifies which plan and preprocessed data to use for training on the given task. Defaults to the YuccaPlanne folder
  --disable_logging     disable logging.
  --ds                  Used to enable deep supervision
  --epochs EPOCHS       Used to specify the number of epochs for training. Default is 1000
  --experiment EXPERIMENT
                        A name for the experiment being performed, with no spaces.
  --loss LOSS           Should only be used to employ alternative Loss Function
  --lr LR               Should only be used to employ alternative Learning Rate.
  --mom MOM             Should only be used to employ alternative Momentum.
  --new_version         Start a new version, instead of continuing from the most recent.
  --patch_size PATCH_SIZE
                        Use your own patch_size. Example: if 32 is provided and the model is 3D we will use patch size (32, 32, 32). Can also be min, max or mean.
  --precision PRECISION
  --profile             Enable profiling.
  --split_idx SPLIT_IDX
                        idx of splits to use for training.
  --split_data_method SPLIT_DATA_METHOD
                        Specify splitting method. Either kfold, simple_train_val_split
  --split_data_param SPLIT_DATA_PARAM
                        Specify the parameter for the selected split method. For KFold use an int, for simple_split use a float between 0.0-1.0.
  --train_batches_per_step TRAIN_BATCHES_PER_STEP
  --val_batches_per_step VAL_BATCHES_PER_STEP
```

**-t**, **-d**, **-m**, **-man** and **-pl** are covered in the front page [ReadMe](/yucca/README.md#training).

**--disable_logging**: Disables logging to WandB and a local logfile. The hparams.yaml will still be saved for experiment inspection and reproducibility.
**--ds**: Enables deep supervision. Note that not all models support this. Currently supported is `UNet`.
**--epochs**: Used to change the number of epochs to train for. 
**--experiment**: Used to change the name of the experiment. This can be useful to sort runs in WandB by specific keys.
**--loss**: Used to change the loss functions. This is mainly intended to be for experimental purposes. For permanent solutions create a specific Manager for the experiment.
**--lr**: Used to change the learning rate. This is mainly intended to be for experimental purposes. For permanent solutions create a specific Manager for the experiment.
**--mom**: Used to change the momentum. This is mainly intended to be for experimental purposes. For permanent solutions create a specific Manager for the experiment.
**--patch_size**: Used to manually specify a patch size rather than letting Yucca calculate a suitable patch size for the dataset.
**--precision**: Used to specify which precision the model should be trained with.
**--profile**: Used to enable profiling for more in-depth debugging.
**--split_idx**: Used to specify which fold/split to use for training. This is almost exclusively used for K-Fold splits.
**--split_data_method**: Used to specify which splitting method to use for training. 
**--split_data_param**: Used to specify the param used for the split method.
**--train_batches_per_step**: Used to specify how many batches are trained on per step. Defaults to 250.
**--train_batches_per_step**: Used to specify how many batches are validated on per step. Defaults to 50.

An example of training on a task called `Task002_NotBrains`, using a 2D `MultiResUnet` with the `YuccaManager_NoPatches` and `YuccaPlanner_224x224` on fold 3 of a 5-Fold split:
```
> yucca_train -t Task002_NotBrains -m MultiResUNet -d 2D -man YuccaManager_NoPatches -pl YuccaPlanner_224x224 -f 3
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
  -tr TR           Full name of Trainer Class. e.g. 'YuccaTrainer_DCE' or 'YuccaManager'. Defaults to YuccaManager.
  -pl PL           Plan ID. Defaults to YuccaPlannerV2
  -chk CHK         Checkpoint to use for inference. Defaults to checkpoint_best.
  --ensemble       Used to initialize data preprocessing for ensemble/2.5D training
  --not_strict     Strict determines if all expected modalities must be present, with the appropriate suffixes (e.g. '_000.nii.gz'). Only touch if you know what you're doing.
  --save_softmax   Save softmax outputs. Required for softmax fusion.
  --overwrite      Overwrite existing predictions
  --no_eval        Disable evaluation and creation of metrics file (result.json)
  --predict_train  Predict on the training set. Useful for debugging.
```
