import os
from yucca.modules.networks.networks import TinyUNet
from yucca.modules.optimization.loss_functions.nnUNet_losses import DiceCE
from yucca.paths import (
    get_models_path,
    get_results_path,
    get_preprocessed_data_path,
    get_raw_data_path,
)

model = TinyUNet
classes = [0, 1]
modalities = ("MRI",)

config = {
    "batch_size": 2,
    "classes": classes,
    "config_name": "demo",
    "crop_to_nonzero": True,
    "continue_from_most_recent": True,
    "deep_supervision": False,
    "experiment": "default",
    "extension": ".nii.gz",
    "learning_rate": 1e-3,
    "loss_fn": DiceCE,
    "max_epochs": 2,
    "modalities": modalities,
    "model_dimensions": "2D",
    "model": model,
    "model_name": model.__name__,
    "momentum": 0.99,
    "norm_op": "volume_wise_znorm",
    "num_classes": len(classes),
    "num_modalities": len(modalities),
    "patch_size": (32, 32),
    "plans": None,
    "split_idx": 0,
    "split_method": "kfold",
    "split_param": 5,
    "target_size": None,
    "target_spacing": [1.0, 1.0, 1.0],
    "target_coordinate_system": "RAS",
    "task": "Task000_TEST_SEGMENTATION",
    "task_type": "segmentation",
}


ckpt_path = os.path.join(
    get_models_path(),
    config["task"],
    config["model_name"] + "__" + config["model_dimensions"],
    "__" + config["config_name"],
    "default",
    "kfold_5_fold_0",
    "version_0",
    "checkpoints",
    "last.ckpt",
)

inference_save_path = os.path.join(
    get_results_path(),
    config["task"],
    config["task"],
    config["model_name"] + "__" + config["model_dimensions"],
    "__" + config["config_name"],
    "kfold_5_fold_0",
    "version_0",
    "best",
)
