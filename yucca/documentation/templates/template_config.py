config = {
    "allow_missing_modalities": False,
    "batch_size": 2,
    "crop_to_nonzero": True,
    "dims": "2D",
    "deep_supervision": False,
    "experiment": "default",
    "extension": ".nii.gz",
    "learning_rate": 1e-3,
    "loss_fn": "DiceCE",
    "modalities": ("MRI",),
    "model_name": "TinyUNet",
    "momentum": 0.99,
    "norm_op": "volume_wise_znorm",
    "num_classes": 3,
    "num_modalities": 1,
    "patch_size": (32, 32),
    "plans_name": "demo",
    "plans": None,
    "split_idx": 0,
    "split_method": "kfold",
    "split_param": 5,
    "task": "Task001_OASIS",
    "task_type": "segmentation",
}
