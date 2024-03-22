from torchvision import transforms
from yucca.image_processing.matrix_ops import get_max_rotated_size
from yucca.image_processing.transforms.formatting import (
    AddBatchDimension,
    RemoveBatchDimension,
)
from yucca.image_processing.transforms.BiasField import BiasField
from yucca.image_processing.transforms.Blur import Blur
from yucca.image_processing.transforms.CopyImageToSeg import CopyImageToSeg
from yucca.image_processing.transforms.Gamma import Gamma
from yucca.image_processing.transforms.Ghosting import MotionGhosting
from yucca.image_processing.transforms.Masking import Masking
from yucca.image_processing.transforms.Mirror import Mirror
from yucca.image_processing.transforms.Noise import (
    AdditiveNoise,
    MultiplicativeNoise,
)
from yucca.image_processing.transforms.Ringing import GibbsRinging
from yucca.image_processing.transforms.sampling import DownsampleSegForDS
from yucca.image_processing.transforms.SimulateLowres import SimulateLowres
from yucca.image_processing.transforms.Spatial import Spatial


class YuccaAugmentationComposer:
    def __init__(
        self,
        patch_size: list | tuple,
        deep_supervision: bool = False,
        is_2D: bool = False,
        parameter_dict: dict = {},
        task_type_preset: str = "segmentation",
    ):
        self._pre_aug_patch_size = None
        self.deep_supervision = deep_supervision
        self.setup_default_params(is_2D, patch_size)
        self.apply_task_type_specific_preset(task_type_preset)
        self.overwrite_params(parameter_dict)
        self.train_transforms = self.compose_train_transforms()
        self.val_transforms = self.compose_val_transforms()

    def setup_default_params(self, is_2D, patch_size):
        print("Composing Transforms")
        # Define whether we crop before or after applying augmentations
        # Define if the cropping performed during the spatial transform is random or always centered
        #   We have ALREADY selected a random crop when we prepared the case in the dataloader
        #   so unless you know what you're doing this should be disabled to avoid border artifacts
        self.random_crop = False
        self.mask_image_for_reconstruction = False
        self.patch_size = patch_size
        self.cval = "min"  # can be an int, float or a str in ['min', 'max']

        # label/segmentation transforms
        self.skip_label = False
        self.label_dtype = int
        self.copy_image_to_label = False

        # Default probabilities (all ON by default)
        # For augmentations with AUG_P_PER_SAMPLE, AUG_P_PER_CHANNEL and AUG_P_PER_AXIS they are applied in the following order:
        #
        # for sample in dataset: # iterate over the batch
        #  if p_per_sample > np.random.uniform():
        #    for channel in sample: # iterate over the channel/modalities
        #      if p_per_channel > np.random.uniform():
        #        for axis in channel: # iterate over the h,w,d dimensions
        #          if p_per_axis > np.random.uniform():
        #            sample[channel, axis] = apply_aug(sample[channel, axis])
        # Therefore, to enable any augmentation the AUG_P_PER_SAMPLE must be > 0.

        self.additive_noise_p_per_sample = 0.2
        self.biasfield_p_per_sample = 0.33
        self.blurring_p_per_sample = 0.2
        self.blurring_p_per_channel = 0.5
        self.elastic_deform_p_per_sample = 0.33
        self.gamma_p_per_sample = 0.2
        self.gamma_p_invert_image = 0.05
        self.gibbs_ringing_p_per_sample = 0.2
        self.mirror_p_per_sample = 0.0
        self.mirror_p_per_axis = 0.33
        self.motion_ghosting_p_per_sample = 0.2
        self.multiplicative_noise_p_per_sample = 0.2
        self.rotation_p_per_sample = 0.2
        self.rotation_p_per_axis = 0.66
        self.scale_p_per_sample = 0.2
        self.simulate_lowres_p_per_sample = 0.2
        self.simulate_lowres_p_per_channel = 0.5
        self.simulate_lowres_p_per_axis = 0.33

        # default augmentation values
        self.additive_noise_mean = (0.0, 0.0)
        self.additive_noise_sigma = (1e-3, 1e-4)

        self.blurring_sigma = (0.0, 1.0)

        self.elastic_deform_alpha = (200, 600)
        self.elastic_deform_sigma = (20, 30)

        self.gamma_range = (0.5, 2.0)

        self.gibbs_ringing_cutfreq = (96, 129)
        self.gibbs_ringing_axes = (0, 2) if is_2D else (0, 3)

        self.mask_ratio = 0.5

        self.mirror_axes = (0, 1) if is_2D else (0, 1, 2)

        self.motion_ghosting_alpha = (0.85, 0.95)
        self.motion_ghosting_numreps = (2, 11)
        self.motion_ghosting_axes = (0, 2) if is_2D else (0, 3)

        self.multiplicative_noise_mean = (0, 0)
        self.multiplicative_noise_sigma = (1e-3, 1e-4)

        self.rotation_x = (-30.0, 30.0)
        self.rotation_y = (-0.0, 0.0) if is_2D else (-30.0, 30.0)
        self.rotation_z = (-0.0, 0.0) if is_2D else (-30.0, 30.0)

        self.scale_factor = (0.9, 1.1)

        self.simulate_lowres_zoom_range = (0.5, 1.0)

    @property
    def pre_aug_patch_size(self):
        # First check if any spatial transforms are included
        if self.elastic_deform_p_per_sample > 0 or self.rotation_p_per_sample > 0 or self.scale_p_per_sample > 0:
            self._pre_aug_patch_size = get_max_rotated_size(self.patch_size)
        return self._pre_aug_patch_size

    def apply_task_type_specific_preset(self, preset):
        if preset == "segmentation":
            # we do nothing and use the default parameters
            pass

        elif preset == "classification":
            self.skip_label = True

        elif preset == "self-supervised":
            self.skip_label = True
            self.copy_image_to_label = True
            self.mask_image_for_reconstruction = True

        else:
            raise ValueError(f"{preset} is not a valid `task_type_preset`.")

    def overwrite_params(self, parameter_dict):
        for key, value in parameter_dict.items():
            setattr(self, key, value)

    def compose_train_transforms(self):
        tr_transforms = transforms.Compose(
            [
                AddBatchDimension(),
                Spatial(
                    patch_size=self.patch_size,
                    crop=True,
                    random_crop=self.random_crop,
                    cval=self.cval,
                    p_deform_per_sample=self.elastic_deform_p_per_sample,
                    deform_sigma=self.elastic_deform_sigma,
                    deform_alpha=self.elastic_deform_alpha,
                    p_rot_per_sample=self.rotation_p_per_sample,
                    p_rot_per_axis=self.rotation_p_per_axis,
                    x_rot_in_degrees=self.rotation_x,
                    y_rot_in_degrees=self.rotation_y,
                    z_rot_in_degrees=self.rotation_z,
                    p_scale_per_sample=self.scale_p_per_sample,
                    scale_factor=self.scale_factor,
                    skip_label=self.skip_label,
                ),
                AdditiveNoise(
                    p_per_sample=self.additive_noise_p_per_sample,
                    mean=self.additive_noise_mean,
                    sigma=self.additive_noise_sigma,
                ),
                Blur(
                    p_per_sample=self.blurring_p_per_sample,
                    p_per_channel=self.blurring_p_per_channel,
                    sigma=self.blurring_sigma,
                ),
                MultiplicativeNoise(
                    p_per_sample=self.multiplicative_noise_p_per_sample,
                    mean=self.multiplicative_noise_mean,
                    sigma=self.multiplicative_noise_sigma,
                ),
                MotionGhosting(
                    p_per_sample=self.motion_ghosting_p_per_sample,
                    alpha=self.motion_ghosting_alpha,
                    numReps=self.motion_ghosting_numreps,
                    axes=self.motion_ghosting_axes,
                ),
                GibbsRinging(
                    p_per_sample=self.gibbs_ringing_p_per_sample,
                    cutFreq=self.gibbs_ringing_cutfreq,
                    axes=self.gibbs_ringing_axes,
                ),
                SimulateLowres(
                    p_per_sample=self.simulate_lowres_p_per_sample,
                    p_per_channel=self.simulate_lowres_p_per_channel,
                    p_per_axis=self.simulate_lowres_p_per_axis,
                    zoom_range=self.simulate_lowres_zoom_range,
                ),
                BiasField(p_per_sample=self.biasfield_p_per_sample),
                Gamma(
                    p_per_sample=self.gamma_p_per_sample,
                    p_invert_image=self.gamma_p_invert_image,
                    gamma_range=self.gamma_range,
                ),
                Mirror(
                    p_per_sample=self.mirror_p_per_sample,
                    axes=self.mirror_axes,
                    p_mirror_per_axis=self.mirror_p_per_axis,
                    skip_label=self.skip_label,
                ),
                DownsampleSegForDS(deep_supervision=self.deep_supervision),
                CopyImageToSeg(copy=self.copy_image_to_label),
                Masking(mask=self.mask_image_for_reconstruction, pixel_value=self.cval, ratio=self.mask_ratio),
                RemoveBatchDimension(),
            ]
        )
        return tr_transforms

    def compose_val_transforms(self):
        val_transforms = transforms.Compose(
            [
                AddBatchDimension(),
                CopyImageToSeg(copy=self.copy_image_to_label),
                Masking(mask=self.mask_image_for_reconstruction, pixel_value=self.cval, ratio=self.mask_ratio),
                RemoveBatchDimension(),
            ]
        )
        return val_transforms

    def lm_hparams(self):
        hparams = {
            "augmentation_parameters": {
                "deep_supervision": self.deep_supervision,
                "pre_aug_patch_size": self.pre_aug_patch_size,
                "random_crop": self.random_crop,
                "cval": self.cval,
                "mask_image_for_reconstruction": self.mask_image_for_reconstruction,
                "patch_size": self.patch_size,
                "skip_label": self.skip_label,
                "label_dtype": self.label_dtype,
                "copy_image_to_label": self.copy_image_to_label,
                "additive_noise_p_per_sample": self.additive_noise_p_per_sample,
                "additive_noise_mean": self.additive_noise_mean,
                "additive_noise_sigma": self.additive_noise_sigma,
                "biasfield_p_per_sample": self.biasfield_p_per_sample,
                "blurring_p_per_sample": self.blurring_p_per_sample,
                "blurring_sigma": self.blurring_sigma,
                "blurring_p_per_channel": self.blurring_p_per_channel,
                "elastic_deform_p_per_sample": self.elastic_deform_p_per_sample,
                "elastic_deform_alpha": self.elastic_deform_alpha,
                "elastic_deform_sigma": self.elastic_deform_sigma,
                "gamma_p_per_sample": self.gamma_p_per_sample,
                "gamma_p_invert_image": self.gamma_p_invert_image,
                "gamma_range": self.gamma_range,
                "gibbs_ringing_p_per_sample": self.gibbs_ringing_p_per_sample,
                "gibbs_ringing_cutfreq": self.gibbs_ringing_cutfreq,
                "gibbs_ringing_axes": self.gibbs_ringing_axes,
                "mask_ratio": self.mask_ratio,
                "mirror_p_per_sample": self.mirror_p_per_sample,
                "mirror_p_per_axis": self.mirror_p_per_axis,
                "mirror_axes": self.mirror_axes,
                "motion_ghosting_p_per_sample": self.motion_ghosting_p_per_sample,
                "motion_ghosting_alpha": self.motion_ghosting_alpha,
                "motion_ghosting_numreps": self.motion_ghosting_numreps,
                "motion_ghosting_axes": self.motion_ghosting_axes,
                "multiplicative_noise_p_per_sample": self.multiplicative_noise_p_per_sample,
                "multiplicative_noise_mean": self.multiplicative_noise_mean,
                "multiplicative_noise_sigma": self.multiplicative_noise_sigma,
                "rotation_p_per_sample": self.rotation_p_per_sample,
                "rotation_p_per_axis": self.rotation_p_per_axis,
                "rotation_x": self.rotation_x,
                "rotation_y": self.rotation_y,
                "rotation_z": self.rotation_z,
                "scale_p_per_sample": self.scale_p_per_sample,
                "scale_factor": self.scale_factor,
                "simulate_lowres_p_per_sample": self.simulate_lowres_p_per_sample,
                "simulate_lowres_p_per_channel": self.simulate_lowres_p_per_channel,
                "simulate_lowres_p_per_axis": self.simulate_lowres_p_per_axis,
                "simulate_lowres_zoom_range": self.simulate_lowres_zoom_range,
            }
        }
        return hparams


if __name__ == "__main__":
    from yucca.training.augmentation.augmentation_presets import basic

    x = YuccaAugmentationComposer(patch_size=(32, 32), parameter_dict=basic)
    print("ALL AUGMENTATION PARAMETERS: ", x.lm_hparams())
    print("")
    print("BASIC PARAMETERS: ", basic)
