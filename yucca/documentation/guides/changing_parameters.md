# Table of Contents
- [Guide to Changing Yucca Parameters](#guide-to-changing-yucca-parameters)
- [Subclassing](#subclassing)
- [Preprocessing](#preprocessing)
  * [Spacing](#size-or-spacing)
  * [Orientation](#orientation)
  * [Normalization](#normalization)
- [Training](#training)
  * [Data Augmentation](#data-augmentation)
  * [Data Splits](#data-splits)
  * [Learning Rate](#learning-rate)
  * [Learning Rate Scheduler](#learning-rate-scheduler)
  * [Loss Function](#loss-function)
  * [Model Architecture](#model-architecture)
  * [Model Dimensionality](#model-dimensionality)
  * [Momentum](#momentum)
  * [Optimizer](#optimizer)
- [Inference](#inference)
  * [Evaluation](#evaluation)
  * [Fusion](#fusion)


# Guide to Changing Yucca Parameters
## Subclassing
Changing parameters in Yucca is generally achieved using subclasses. This means, to change a given parameter (1) Subclass the class defining the parameter, (2) change the value of the parameter to the desired value and (3) create a new .py file in the (sub)directory of the parent class.

E.g. to lower the starting Learning Rate we subclass YuccaManager - the default class responsible for handling model training, and change the variable self.learning_rate variable from 1e-3 to 1e-5.

We call this new Manager "YuccaManager_1e5" and save it as "YuccaManager_1e5.py" in the /yucca/training/managers directory as this is where the Parent Class is located. Alternatively it can be saved in a subdirectory of the directory of the parent class e.g. /yucca/training/managers/lr. 

```
from yucca.managers.YuccaManager import YuccaManager

class YuccaManager_1e5(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(learning_rate=learning_rate, *args, **kwargs)
    self.learning_rate = 1e-5

```

# Preprocessing
Unless otherwise mentioned, preprocessing variables and functions are handled by the YuccaPlanners. For optimal results, it is advisable to subclass the default planner when applying changes.

**Default Planner Class: [YuccaPlanner](/yucca/planning/YuccaPlanner.py)**

## Size OR Spacing
Parent: default planner class

Function: *determine_target_size_from_fixed_size_or_spacing*

Either a fixed target size or a fixed target spacing can be specified. By default the preprocessor will set the fixed target spacing to the median spacing of the dataset but this can be overriden if a fixed spacing is desired. To facilitate training on full images (as opposed to the default patch-based training) a fixed target size must be used. 

To set the fixed target spacing to [0.5, 0.5, 1.]:
```
def determine_target_size_from_fixed_size_or_spacing(self):
    self.fixed_target_size = None
    self.fixed_target_spacing = [0.5, 0.5, 1.]
```

To set the fixed target size to the size of the largest image of the original/unprocessed dataset:
```
def determine_target_size_from_fixed_size_or_spacing(self):
    self.fixed_target_size = self.dataset_properties["original_max_size"]
    self.fixed_target_spacing = None
```

## Orientation
Parent: default planner class

Variable 1: *self.target_coordinate_system*
- used to ensure uniform starting orientations. This is applied before any transposition. NOTE: This will only be applied to samples with valid nifti headers (either using qform or sform codes).

Variable 2: *self.transpose_forward* 
- used to transpose samples from the starting position to a target orientation.

Variable 3: *self.transpose_backward*
- used to transpose samples back from the target orientation to the starting position. This should only be used if samples are transposed by *transpose_forward*. This is applied during inference to revert any transform applied by *transpose_forward*.

For example, if it is desired to first reorient all NIFTI samples to 'LPS' as the starting position, and then transpose them from [h, w, d] to [d, h, w] during training, and finally back from [d, h, w] to [h, w, d] in inference do:

```
self.target_coordinate_system = 'LPS'
self.transpose_forward = [2, 0, 1]
self.transpose_backward = [1, 2, 0]
```

By default *transpose_forward* and *transpose_backward* are both = [0, 1, 2] which means samples are NOT transposed.
This means 2D models are trained on the sagittal view (because images are always sliced in the first dimension, i.e. image[slice, :, :]).
The YuccaPlannerY and YuccaPlannerZ are existing YuccaPlanner implementations designed to train models on the Coronal and Axial views, respectively.

## Normalization
Parent: default planner class

Variable: *self.norm_op* 

To find currently implemented normalization operations see the [normalizer function](/yucca/preprocessing/normalization.py). 

For example if 'minmax' (otherwise known as 0-1 normalization) is desired:
```
self.norm_op = 'minmax'
```

# Training
Unless otherwise mentioned, training variables and functions are handled by the YuccaManagers. For optimal results, it is advisable to subclass the default class when applying changes.

**Default Manager Class: [YuccaManager](/yucca/training/managers/YuccaManager.py)**

## Data Augmentation
Parent: default Manager class

Variable: *self.augmentation_params*

Changing the data augmentation parameters is achieved by defining a dictionary of augmentation parameters in the Manager, which will then automatically apply these settings to the composed augmentations. The default augmentation parameters can be found in the `setup_default_params` method of the [`YuccaAugmentationComposer`](/yucca/training/augmentation/YuccaAugmentationComposer.py). Most augmentations have a variable called "X_p_per_sample" with a floating point value between 0.0-1.0 which controls the probability they are applied to each sample. To disable an augmentation set this probability to 0.0. Some augmentations also have variables that control the possible intensity ranges of the augmentation, such as the `"rotation_x": (-30.0, 30.0)` specifying the minimum and maximum degree of rotation around the X-axis.


To disable blurring entirely and modify the scaling range do:
```
from yucca.managers.YuccaManager import YuccaManager

class YuccaManager_NewUserSetup(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_parameters = {"blurring_p_per_sample": 0.0,
                                        "scale_factor": (0.7, 1.3)}
```

## Data Splits
CLI: `yucca_train`, `yucca_finetune` and `yucca_inference`

Variables: *--split_idx*, *--split_data_method* and *--split_data_param*

NOTE: Do not confuse this with the train-test splits. These must be handled in [Task Conversion](/yucca/documentation/guides/task_conversion.md).

Training/Validation data splits can be automatically generated using simple dataset splits or the K-Fold method. First select the method with `--split_data_method` and then supply the desired parameter with `--split_data_param`. If the method is `simple_train_val_split` valid parameter values are between 0.0-1.0, so a valid configuration could be `--split_data_method simple_train_val_split --split_data_param 0.33`. For `kfold` valid parameters are integers > 1. When using K-Fold the `--split_idx` can be used to select which fold to use for training. When using `split_data_method simple_train_val_split` you do not need to specify the `--split_idx` as there will only be one split of the specified configuration.  By default Yucca will generate 5-Folds and train on Fold 0. 

Newly generated splits are saved in the `splits.pkl` file next to the preprocessed dataset. If this file already exists Yucca will reuse splits generated with the same configuration. To reuse a split previously generated using `--split_data_method simple_train_val_split --split_data_param 0.33` simply specify `--split_data_method simple_train_val_split --split_data_param 0.33` again. To reuse the 5 folds generated with `--split_data_kfold 5` but train on a fold 1 instead of 0 simply specify `--split_data_kfold 5` again, but this time with `--split_idx 1`.


This means, for any dataset it's partitioned into five parts [0, 1, 2, 3, 4], which we use to create 5 splits:
```
{
'kfold': 
    {'5': 
        [
        {'train': [1, 2, 3, 4], 'val': [0]}, 
        {'train': [0, 2, 3, 4], 'val': [1]}, 
        {'train': [0, 1, 3, 4], 'val': [2]}, 
        {'train': [0, 1, 2, 4], 'val': [3]}, 
        {'train': [0, 1, 2, 3], 'val': [4]}, 
        ]
    }
'simple_train_val_split:
    {'0.40': 
        [
        {'train': [0, 2, 4], 'val': [1, 3]}
        ]
    }
}
```

If you wish to use predefined splits, you have to manufacture a splits file and save it (or append to an existing file) in the folder of the task's preprocessed data (e.g. "/path/to/YuccaData/yucca_preprocessed/TaskXXX_MyTask/splits.pkl").

When doing this, the contents of the manufactured split file should be a list containing a dictionary:
```
{
'mycustomsplit':
    {'version0':
        [
        {'train': [0], 'val': [1, 2, 3, 4]}
        ]
    }
}
```
Which is then selected using `--split_data_method custom --split_data_param version0`

## Deep Supervision
CLI: In training and finetuning deep supervision is enabled using the --ds flag. 

## Learning Rate
Parent: default Manager class

Variable: self.learning\_rate

```
from yucca.managers.YuccaManager import YuccaManager

class YuccaManager_LowerLR(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = 1e-5
```

CLI: Can also be changed using --lr flag.

## Learning Rate Scheduler
Parent: [YuccaLightningModule](/yucca/training/lightning_modules/YuccaLightningModule.py)

Variable: self.lr\_scheduler
- Used to determine the LR Scheduler CLASS

```
from yucca.lightning_modules.YuccaLightningModule import YuccaLightningModule
from torch import optim

class YuccaLightningModule_StepLRS(YuccaLightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_scheduler = optim.lr_scheduler.StepLR
```

For an illustration of how a wide selection of Torch.Optim LR Schedulers work see the [LR Scheduler Illustration Notebook](/yucca/documentation/illustrations/learning_rate_schedulers.ipynb).
The notebook illustrates the effect of the scheduling algorithms with different parameters. 

## Loss Function
Parent: default Manager class

Variable: self.loss

The loss class must be saved in /yucca/training/loss_and_optim/loss_functions and be a subclass of nn.Module.

```
from yucca.managers.YuccaManager import YuccaManager
from yucca.loss_and_optim.loss_functions import NLL

class YuccaManager_NLL(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = NLL
```

Can also be changed using *yucca_train* --loss flag.

## Model Architecture
CLI: In both training and inference model architecture is specified using the -m flag. Currently supported architectures can be found in [networks](/yucca/network_architectures/networks).

## Model Dimensionality
CLI: In both preprocessing, training and inference dimension is specified using the -d flag. Currently supported is: "2D" and "3D".

## Momentum
Parent: default Manager class

Variable: self.momentum 

```
from yucca.managers.YuccaManager import YuccaManager

class YuccaManager_mom95(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum= 0.95
```

CLI: Can also be changed using *yucca_train* --mom flag.

## Optimizer
Parent: [YuccaLightningModule](/yucca/training/lightning_modules/YuccaLightningModule.py)

Variable: self.optim

```
from yucca.lightning_modules.YuccaLightningModule import YuccaLightningModule
from torch import optim

class YuccaLightningModule_Adam(YuccaLightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim = optim.Adam
```

## Patch Based Training
Parent: default Manager class

Variable: self.patch_based_training 

```
from yucca.managers.YuccaManager import YuccaManager

class YuccaManager_NoPatches(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_based_training = False
```

This is used to train models on full-size images. Requires datasets are preprocessed using a Planner using fixed_target_size to ensure that all samples have identical dimensions.

# Inference
Changing inference parameters not implemented currently.

## Evaluation
## Pixel/Object
## Fusion
