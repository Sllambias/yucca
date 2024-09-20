from yucca.modules.networks.networks import TinyUNet
from yucca.modules.optimization.loss_functions.nnUNet_losses import DiceCE

model = TinyUNet

config = {
    "batch_size": 2,
    "classes": [0, 1],
    "config_name": "demo",
    "crop_to_nonzero": True,
    "continue_from_most_recent": True,
    "deep_supervision": False,
    "experiment": "default",
    "extension": ".nii.gz",
    "learning_rate": 1e-3,
    "loss_fn": DiceCE,
    "max_epochs": 2,
    "modalities": ("MRI",),
    "model_dimensions": "2D",
    "model": TinyUNet,
    "momentum": 0.99,
    "norm_op": "volume_wise_znorm",
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

config["model_name"] = config["model"].__name__
config["num_classes"] = len(config["classes"])
config["num_modalities"] = len(config["modalities"])
