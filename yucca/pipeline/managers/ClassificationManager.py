from yucca.pipeline.managers.YuccaManagerV2 import YuccaManagerV2
from yucca.modules.data.augmentation.augmentation_presets import genericV2
from yucca.modules.lightning_modules.ClassificationLightningModule import ClassificationLightningModule
from yucca.modules.data.datasets.ClassificationDataset import ClassificationTrainDataset, ClassificationTestDataset
from yucca.modules.optimization.loss_functions.CE import CE


class ClassificationManagerV2(YuccaManagerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = genericV2
        self.augmentation_params["skip_label"] = True
        self.model_name = "resnet18"
        self.loss = CE
        self.lightning_module = ClassificationLightningModule
        self.model_dimensions = "3D"
        self.patch_based_training = False
        self.deep_supervision = False
        self.train_dataset_class = ClassificationTrainDataset
        self.test_dataset_class = ClassificationTestDataset


class ClassificationManagerV9(ClassificationManagerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = self.set_aug_params()
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
        from yucca.modules.data.augmentation.augmentation_presets import channel_specific_probas

        self.augmentation_params.update(channel_specific_probas)
