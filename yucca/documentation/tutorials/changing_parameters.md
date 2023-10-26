# Table of Contents
- [Guide to Changing Yucca Parameters](#guide-to-changing-yucca-parameters)
- [Subclassing](#subclassing)
- [Preprocessing](#preprocessing)
  * [Spacing](#spacing)
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
Changing parameters in Yucca is generally achieved using subclasses. (1) Subclass the class defining the parameter of interest, (2) change the variable to the desired value and (3) create a new .py file in the (sub)directory of the parent class.

E.g. to lower the starting Learning Rate we subclass YuccaTrainerV2 - the default class responsible for handling model training, and change the variable \_DEFAULT\_STARTING\_LR variable from 1e-3 to 1e-5.

We call this new trainer "YuccaTrainerV2_LowerLR" and save it as "YuccaTrainerV2_LowerLR.py" in the /yucca/training/trainers directory as this is where the Parent Class is located. Alternatively it can be saved in a subdirectory of the directory of the parent class e.g. /yucca/training/trainers/lr. 

```
from yucca.training.trainers.YuccaTrainerV2 import YuccaTrainerV2

class YuccaTrainerV2_1e5(YuccaTrainerV2):
    def __init__(self, model, model_dimensions: str, task: str, folds: str | int, plan_id: str,
                 starting_lr: float = None, loss_fn: str = None, momentum: float = None,
                 continue_training: bool = False):
        super().__init__(model, model_dimensions, task, folds, plan_id, starting_lr, 
                         loss_fn, momentum, continue_training)
        self._DEFAULT_STARTING_LR = 1e-5
        
```

# Preprocessing
Unless otherwise mentioned, preprocessing variables and functions are handled by the YuccaPlanners. For optimal results, it is advisable to subclass the default planner when applying changes.

**Default Planner Class: [YuccaPlannerV2](/yucca/planning/YuccaPlannerV2.py)**

## Spacing
Parent: default planner class

Function: *determine_spacing*

E.g. if a fixed spacing of [0.5, 0.5, 1.] is desired:

```
def determine_spacing(self):
    return [0.5, 0.5, 1.]
```

## Orientation
Parent: default planner class

Variable 1: *self.target_coordinate_system*
- used to ensure uniform starting orientations. This is applied before any transposition. NOTE: This will only be applied to samples with valid nifti headers (either using qform or sform codes).

Variable 2: *self.transpose_forward* 
- used to transpose samples from the starting position to a target orientation.

Variable 3: *self.transpose_backward*
- used to transpose samples back from the target orientation to the starting position. This should only be used if samples are transposed by *transpose_forward*. This is applied during inference to revert any transform applied by *transpose_forward*.

For example, if it is desired to first reorient all samples to 'LPS' as the starting position, and then transpose them from [x, y, z] to [z, x, y] during training so it effectively becomes SLP, and finally back from [z, x, y] to [x, y, z] in inference do:
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
Unless otherwise mentioned, preprocessing variables and functions are handled by the YuccaTrainers. For optimal results, it is advisable to subclass the default class when applying changes.

**Default Trainer Class: [YuccaTrainerV2](/yucca/training/trainers/YuccaTrainerV2.py)**

## Data Augmentation
Parent: default trainer class

Method: *setup_DA*
Variable: *self.augmentation_parameters*

Changing the data augmentation scheme can be done in two ways. The first method is suitable for minor changes. The second is suitable for major changes.

To make minor changes you should subclass the parent and redefine the *setup_DA* method. Use the super() function to call the parent *setup_DA* function first, and then change the appropriate parameters.

```
def setup_DA(self):
    super().setup_DA()
    self.augmentation_parameters["do_ElasticDeform"] = False
```

Refer to the [default_augmentation_params](/yucca/training/augmentation/default_augmentation_params.py) for all valid arguments.

To make major changes you should (1) subclass the parent and redefine the *setup_DA* method and (2) create a new dictionary of augmentation parameters with the same keys as the [default_augmentation_params](/yucca/training/augmentation/default_augmentation_params.py) (3) change the desired values (4) save this as a new .py file in the /yucca/training/augmentation folder (5) import and assign your new dictionary to *self.augmentation_parameters* in your subclassed trainer:

```
from yucca.training.augmentation.my_new_params import MyNew_3D_params, MyNew_2D_params

def setup_DA(self):
    if self.model_dimensions == '3D':
        self.augmentation_parameters = MyNew_3D_params
    if self.model_dimensions in ['2D', '25D']:
        self.augmentation_parameters = MyNew_2D_params
```

## Data Splits
Parent: default trainer class

Method: *split_data*

NOTE: Do not confuse this with the train-test splits. These must be handled in [Task Conversion](/yucca/documentation/tutorials/task_conversion.md).

By default Training-Validation data splits are created by the *split_data* method the first time a model is trained on a given task. The method randomly partitions the data into 5 equal parts. Subsequently, 5 train-val splits are created by withholding each of the parts as validation data and combining the remaning four parts into training data, equating to a 80%/20% split. Although these splits are randomly generated, each trainer uses a fixed random seed for reproducibility. 

This means, for any dataset it's partitioned into five parts [0, 1, 2, 3, 4], which we use to create 5 splits:
```
[
    {train: [1, 2, 3, 4], val: [0]}, 
    {train: [0, 2, 3, 4], val: [1]}, 
    {train: [0, 1, 3, 4], val: [2]}, 
    {train: [0, 1, 2, 4], val: [3]}, 
    {train: [0, 1, 2, 3], val: [4]}, 
]
```
To specify which split to train on, use the "-f" flag followed by an integer in [0, 4]. By default split 0 ({train: [1, 2, 3, 4], val: [0]}) is used. 

If you wish to use predefined splits, you have to manufacture a splits file and save it in the folder of the task's preprocessed data (e.g. "/path/to/YuccaData/yucca_preprocessed/TaskXXX_MyTask/splits.pkl"). The Trainer will only create the random splits if no "splits.pkl" file is already present in the preprocessed folder. Therefore, manually placing one there will effectively block the Trainer from creating a new splits file.

When doing this, the contents of the manufactured splits file should be a list containing a dictionary:
```
[
    {train: [case1, case2, case3], val: [case4, case5]},
]
```
Which is then selected with the "-f 0" flag in *yucca_train* and *yucca_inference*.

If you wish to change the method you should subclass the parent class and redefine the *split_data* method to use the desired functions or split ratios.

## Learning Rate
Parent: default trainer class

Variable: self.\_DEFAULT\_STARTING\_LR 

Subclass the parent and change the variable to the desired value using scientific notation.

```
from yucca.training.trainers.YuccaTrainerV2 import YuccaTrainerV2

class YuccaTrainerV2_LowerLR(YuccaTrainerV2):
    def __init__(self, model, model_dimensions: str, task: str, folds: str | int, plan_id: str,
                 starting_lr: float = None, loss_fn: str = None, momentum: float = None,
                 continue_training: bool = False):
        super().__init__(model, model_dimensions, task, folds, plan_id, starting_lr, 
                         loss_fn, momentum, continue_training)
        self._DEFAULT_STARTING_LR = 1e-5
```

Can also be changed using *yucca_train* --lr flag.

## Learning Rate Scheduler
For an illustration of how a wide selection of Torch.Optim LR Schedulers work see the [LR Scheduler Illustration Notebook](/yucca/documentation/illustrations/learning_rate_schedulers.ipynb).
The notebook illustrates the effect of the scheduling algorithms with different parameters. 

Parent: default trainer class

Variable 1: self.lr\_scheduler
- Used to determine the LR Scheduler CLASS
Variable 2: self.lr\_scheduler\_kwargs
- Used to supply arguments to the chosen scheduler. key,value pairs irrelevant to the scheduler class will be filtered out automatically.
Subclass the parent and change the variables to the desired values.

```
from yucca.training.trainers.YuccaTrainerV2 import YuccaTrainerV2
from torch import optim

class YuccaTrainerV2_StepLRS(YuccaTrainerV2):
    def __init__(self, model, model_dimensions: str, task: str, folds: str | int, plan_id: str,
                 starting_lr: float = None, loss_fn: str = None, momentum: float = None,
                 continue_training: bool = False):
        super().__init__(model, model_dimensions, task, folds, plan_id, starting_lr, 
                         loss_fn, momentum, continue_training)
        self.lr_scheduler = optim.lr_scheduler.StepLR
        self.lr_scheduler_kwargs = {'step_size':50, 'gamma':0.9}
```

## Loss Function
Parent: default trainer class

Variable: self.\_DEFAULT\_LOSS

Subclass the parent and change the variable to the desired loss class. The loss class must be saved in /yucca/training/loss_functions and be a subclass of nn.Module.

```
from yucca.training.trainers.YuccaTrainerV2 import YuccaTrainerV2
from yucca.training.loss_functions import NLL

class YuccaTrainerV2_NLL(YuccaTrainerV2):
    def __init__(self, model, model_dimensions: str, task: str, folds: str | int, plan_id: str,
                 starting_lr: float = None, loss_fn: str = None, momentum: float = None,
                 continue_training: bool = False):
        super().__init__(model, model_dimensions, task, folds, plan_id, starting_lr, 
                         loss_fn, momentum, continue_training)
        self._DEFAULT_LOSS = NLL
```

Can also be changed using *yucca_train* --loss flag.

## Model Architecture
Changing architecture is entirely handled in the command line. In both training and inference model architecture is specified using the -m flag. Currently supported is: "UNet", "MultiResUnet", "UNetR" and "UXNet".

## Model Dimensionality
Changing model dimensionality is entirely handled in the command line. In both preprocessing, training and inference dimension is specified using the -d flag. Currently supported is: "2D" and "3D".

## Momentum

Parent: default trainer class

Variable: self.\_DEFAULT\_MOMENTUM

Subclass the parent and change the variable to the desired value.

```
from yucca.training.trainers.YuccaTrainerV2 import YuccaTrainerV2

class YuccaTrainerV2_mom95(YuccaTrainerV2):
    def __init__(self, model, model_dimensions: str, task: str, folds: str | int, plan_id: str,
                 starting_lr: float = None, loss_fn: str = None, momentum: float = None,
                 continue_training: bool = False):
        super().__init__(model, model_dimensions, task, folds, plan_id, starting_lr, 
                         loss_fn, momentum, continue_training)
        self._DEFAULT_MOMENTUM= 0.95
```

Can also be changed using *yucca_train* --mom flag.

## Optimizer

Parent: default trainer class

Variable: self.optim

Subclass the parent and change the variable to the desired torch optimizer class.

```
from yucca.training.trainers.YuccaTrainerV2 import YuccaTrainerV2
from torch import optim

class YuccaTrainerV2_Adam(YuccaTrainerV2):
    def __init__(self, model, model_dimensions: str, task: str, folds: str | int, plan_id: str,
                 starting_lr: float = None, loss_fn: str = None, momentum: float = None,
                 continue_training: bool = False):
        super().__init__(model, model_dimensions, task, folds, plan_id, starting_lr, 
                         loss_fn, momentum, continue_training)
        self.optim = optim.Adam
```

# Inference
Changing inference parameters not implemented currently.

## Evaluation
## Pixel/Object
## Fusion
