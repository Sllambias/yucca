import torch
import torch.nn as nn
from yucca.pipeline.managers.YuccaManagerV2 import YuccaManagerV2, YuccaManager
from yucca.modules.data.augmentation.augmentation_presets import genericV2
from yucca.modules.lightning_modules.ClassificationLightningModule import ClassificationLightningModule
from yucca.modules.data.datasets.ClassificationDataset import ClassificationTrainDataset, ClassificationTestDataset
from yucca.modules.optimization.loss_functions.CE import CE


class ClassificationManager(YuccaManagerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = genericV2
        self.augmentation_params["skip_label"] = True
        self.model_name = "ResNet50_Volumetric"
        self.loss = CE
        self.lightning_module = ClassificationLightningModule
        self.model_dimensions = "3D"
        self.patch_based_training = False
        self.deep_supervision = False
        self.train_dataset_class = ClassificationTrainDataset
        self.test_dataset_class = ClassificationTestDataset


class ClassificationManagerV2(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = genericV2
        self.augmentation_params["skip_label"] = True
        self.model_name = "ResNet50_Volumetric"
        self.loss = CE
        self.lightning_module = ClassificationLightningModule
        self.model_dimensions = "3D"
        self.patch_based_training = False
        self.deep_supervision = False
        self.train_dataset_class = ClassificationTrainDataset


class ClassificationManager_ResNet18(ClassificationManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "resnet18"


class ClassificationManagerV2_ResNet18(ClassificationManagerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "resnet18"


class ClassificationManagerV3(YuccaManagerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = self.set_aug_params()
        self.model_name = "resnet18"
        self.loss = CE
        self.lightning_module = ClassificationLightningModule
        self.model_dimensions = "3D"
        self.patch_based_training = False
        self.deep_supervision = False
        self.train_dataset_class = ClassificationTrainDataset

    def set_aug_params(self):
        return {
            "random_crop": True,
            "mask_image_for_reconstruction": False,
            "clip_to_input_range": True,  # ensures no augmentations go beyond the input range of the image/patch
            "normalize": False,
            # label/segmentation transforms
            "skip_label": True,
            "label_dtype": int,
            "copy_image_to_label": False,
            # default augmentation probabilities
            "additive_noise_p_per_sample": 0.3,
            "biasfield_p_per_sample": 0.3,
            "blurring_p_per_sample": 0.3,
            "blurring_p_per_channel": 0.5,
            "elastic_deform_p_per_sample": 0.0,
            "gamma_p_per_sample": 0.3,
            "gamma_p_invert_image": 0.05,
            "gibbs_ringing_p_per_sample": 0.2,
            "mirror_p_per_sample": 0.0,
            "mirror_p_per_axis": 0.33,
            "motion_ghosting_p_per_sample": 0.2,
            "multiplicative_noise_p_per_sample": 0.3,
            "rotation_p_per_sample": 0.33,
            "rotation_p_per_axis": 0.66,
            "scale_p_per_sample": 0.33,
            "simulate_lowres_p_per_sample": 0.3,
            "simulate_lowres_p_per_channel": 0.5,
            "simulate_lowres_p_per_axis": 0.66,
            # default augmentation values
        }


class ClassificationManager_Regularized(ClassificationManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "resnet18"
        self.optim_kwargs["weight_decay"] = 1e-1


class ClassificationManagerV4(YuccaManagerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = self.set_aug_params()
        self.model_name = "resnet18"
        self.loss = CE
        self.lightning_module = ClassificationLightningModule
        self.model_dimensions = "3D"
        self.patch_based_training = False
        self.deep_supervision = False
        self.train_dataset_class = ClassificationTrainDataset
        self.optim_kwargs["weight_decay"] = 1e-1

    def set_aug_params(self):
        return {
            "random_crop": True,
            "mask_image_for_reconstruction": False,
            "clip_to_input_range": True,  # ensures no augmentations go beyond the input range of the image/patch
            "normalize": False,
            # label/segmentation transforms
            "skip_label": True,
            "label_dtype": int,
            "copy_image_to_label": False,
            # default augmentation probabilities
            "additive_noise_p_per_sample": 0.4,
            "biasfield_p_per_sample": 0.4,
            "blurring_p_per_sample": 0.4,
            "blurring_p_per_channel": 0.5,
            "elastic_deform_p_per_sample": 0.2,
            "gamma_p_per_sample": 0.4,
            "gamma_p_invert_image": 0.05,
            "gibbs_ringing_p_per_sample": 0.3,
            "mirror_p_per_sample": 0.0,
            "mirror_p_per_axis": 0.33,
            "motion_ghosting_p_per_sample": 0.3,
            "multiplicative_noise_p_per_sample": 0.4,
            "rotation_p_per_sample": 0.33,
            "rotation_p_per_axis": 0.66,
            "scale_p_per_sample": 0.33,
            "simulate_lowres_p_per_sample": 0.3,
            "simulate_lowres_p_per_channel": 0.5,
            "simulate_lowres_p_per_axis": 0.66,
            # default augmentation values
        }


class ClassificationManagerV5(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = self.set_aug_params()
        self.model_name = "resnet18"
        self.loss = CE
        self.lightning_module = ClassificationLightningModule
        self.model_dimensions = "3D"
        self.patch_based_training = False
        self.deep_supervision = False
        self.train_dataset_class = ClassificationTrainDataset
        self.optim_kwargs["weight_decay"] = 5e-2

    def set_aug_params(self):
        return {
            "random_crop": True,
            "mask_image_for_reconstruction": False,
            "clip_to_input_range": True,  # ensures no augmentations go beyond the input range of the image/patch
            "normalize": False,
            # label/segmentation transforms
            "skip_label": True,
            "label_dtype": int,
            "copy_image_to_label": False,
            # default augmentation probabilities
            "additive_noise_p_per_sample": 0.4,
            "biasfield_p_per_sample": 0.4,
            "blurring_p_per_sample": 0.4,
            "blurring_p_per_channel": 0.5,
            "elastic_deform_p_per_sample": 0.2,
            "gamma_p_per_sample": 0.4,
            "gamma_p_invert_image": 0.05,
            "gibbs_ringing_p_per_sample": 0.3,
            "mirror_p_per_sample": 0.0,
            "mirror_p_per_axis": 0.33,
            "motion_ghosting_p_per_sample": 0.3,
            "multiplicative_noise_p_per_sample": 0.4,
            "rotation_p_per_sample": 0.33,
            "rotation_p_per_axis": 0.66,
            "scale_p_per_sample": 0.33,
            "simulate_lowres_p_per_sample": 0.3,
            "simulate_lowres_p_per_channel": 0.5,
            "simulate_lowres_p_per_axis": 0.66,
            # default augmentation values
        }


class ClassificationManagerV6(ClassificationManagerV5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["weight_decay"] = 1e-1


class ClassificationManagerV7(ClassificationManagerV5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["weight_decay"] = 1e-2


class ClassificationManagerV8(ClassificationManagerV5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["weight_decay"] = 1e-2
        self.augmentation_params["elastic_deform_p_per_sample"] = 0.0


class ClassificationManagerV9(ClassificationManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = self.set_aug_params()
        self.model_name = "resnet18"
        self.optim_kwargs["weight_decay"] = 5e-2

    def set_aug_params(self):
        return {
            "random_crop": True,
            "mask_image_for_reconstruction": False,
            "clip_to_input_range": True,  # ensures no augmentations go beyond the input range of the image/patch
            "normalize": False,
            # label/segmentation transforms
            "skip_label": True,
            "label_dtype": int,
            "copy_image_to_label": False,
            # default augmentation probabilities
            "additive_noise_p_per_sample": 0.4,
            "biasfield_p_per_sample": 0.4,
            "blurring_p_per_sample": 0.4,
            "blurring_p_per_channel": 0.5,
            "elastic_deform_p_per_sample": 0.0,
            "gamma_p_per_sample": 0.4,
            "gamma_p_invert_image": 0.05,
            "gibbs_ringing_p_per_sample": 0.3,
            "mirror_p_per_sample": 0.0,
            "mirror_p_per_axis": 0.33,
            "motion_ghosting_p_per_sample": 0.3,
            "multiplicative_noise_p_per_sample": 0.4,
            "rotation_p_per_sample": 0.33,
            "rotation_p_per_axis": 0.66,
            "scale_p_per_sample": 0.33,
            "simulate_lowres_p_per_sample": 0.3,
            "simulate_lowres_p_per_channel": 0.5,
            "simulate_lowres_p_per_axis": 0.66,
            # default augmentation values
        }


class ClassificationManagerV10(ClassificationManagerV9):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["lr"] = 5e-6


class ClassificationManagerV11(ClassificationManagerV9):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["lr"] = 1e-6


class ClassificationManagerV12(ClassificationManagerV9):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["lr"] = 5e-6
        self.optim_kwargs["weight_decay"] = 1e-2


class ClassificationManagerV13(ClassificationManagerV9):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["lr"] = 1e-6
        self.optim_kwargs["weight_decay"] = 5e-3


class ClassificationManagerV14(ClassificationManagerV9):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["lr"] = 5e-6
        self.optim_kwargs["weight_decay"] = 5e-3


class ClassificationManagerV15(ClassificationManagerV9):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["lr"] = 1e-6
        self.optim_kwargs["weight_decay"] = 1e-2
        self.test_dataset_class = ClassificationTestDataset


class ClassificationManagerV16(ClassificationManagerV9):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "resnet18_dropout"
        self.optim_kwargs["lr"] = 1e-6
        self.optim_kwargs["weight_decay"] = 1e-2


class ClassificationManagerV17(ClassificationManagerV9):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "resnet18_dropout"
        self.optim_kwargs["lr"] = 1e-6


class ClassificationManagerV18(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = self.set_aug_params()
        self.loss = CE
        self.lightning_module = ClassificationLightningModule
        self.model_dimensions = "3D"
        self.patch_based_training = False
        self.deep_supervision = False
        self.train_dataset_class = ClassificationTrainDataset

        self.model_name = "resnet18_dropout"
        self.optim_kwargs["lr"] = 1e-5
        self.optim_kwargs["weight_decay"] = 5e-3

    def set_aug_params(self):
        return {
            "random_crop": True,
            "mask_image_for_reconstruction": False,
            "clip_to_input_range": True,  # ensures no augmentations go beyond the input range of the image/patch
            "normalize": False,
            # label/segmentation transforms
            "skip_label": True,
            "label_dtype": int,
            "copy_image_to_label": False,
            # default augmentation probabilities
            "additive_noise_p_per_sample": 0.3,
            "biasfield_p_per_sample": 0.3,
            "blurring_p_per_sample": 0.3,
            "blurring_p_per_channel": 0.4,
            "elastic_deform_p_per_sample": 0.0,
            "gamma_p_per_sample": 0.3,
            "gamma_p_invert_image": 0.05,
            "gibbs_ringing_p_per_sample": 0.2,
            "mirror_p_per_sample": 0.0,
            "mirror_p_per_axis": 0.33,
            "motion_ghosting_p_per_sample": 0.2,
            "multiplicative_noise_p_per_sample": 0.3,
            "rotation_p_per_sample": 0.2,
            "rotation_p_per_axis": 0.66,
            "scale_p_per_sample": 0.33,
            "simulate_lowres_p_per_sample": 0.2,
            "simulate_lowres_p_per_channel": 0.5,
            "simulate_lowres_p_per_axis": 0.66,
            # default augmentation values
        }


class ClassificationManagerV19(ClassificationManagerV18):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["lr"] = 1e-6


class ClassificationManagerV20(ClassificationManagerV18):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["weight_decay"] = 1e-2


class ClassificationManagerV21(ClassificationManagerV18):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["weight_decay"] = 1e-2
        self.optim_kwargs["lr"] = 1e-6


class ClassificationManagerV9_1(ClassificationManagerV9):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["lr"] = 5e-6
        self.optim_kwargs["weight_decay"] = 1e-2


class ClassificationManagerV9_2(ClassificationManagerV9):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["lr"] = 1e-5
        self.optim_kwargs["weight_decay"] = 1e-2


class ClassificationManagerV9_3(ClassificationManagerV9):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["lr"] = 1e-6
        self.optim_kwargs["weight_decay"] = 5e-3


class ClassificationManagerV9_4(ClassificationManagerV9):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["lr"] = 5e-6
        self.optim_kwargs["weight_decay"] = 5e-3


class ClassificationManagerV22(ClassificationManagerV18):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["lr"] = 1e-3


class ClassificationManagerV23(ClassificationManagerV18):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["lr"] = 1e-4


class ClassificationManagerV24(ClassificationManagerV18):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim_kwargs["lr"] = 1e-5
