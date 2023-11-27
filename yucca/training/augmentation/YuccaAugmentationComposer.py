from torchvision import transforms
from yuccalib.image_processing.matrix_ops import get_max_rotated_size
from yuccalib.image_processing.transforms.formatting import (
    AddBatchDimension,
    RemoveBatchDimension,
)
from yuccalib.image_processing.transforms.BiasField import BiasField
from yuccalib.image_processing.transforms.Blur import Blur
from yuccalib.image_processing.transforms.CopyImageToSeg import CopyImageToSeg
from yuccalib.image_processing.transforms.Gamma import Gamma
from yuccalib.image_processing.transforms.Ghosting import MotionGhosting
from yuccalib.image_processing.transforms.Masking import Masking
from yuccalib.image_processing.transforms.Mirror import Mirror
from yuccalib.image_processing.transforms.Noise import (
    AdditiveNoise,
    MultiplicativeNoise,
)
from yuccalib.image_processing.transforms.Ringing import GibbsRinging
from yuccalib.image_processing.transforms.sampling import DownsampleSegForDS
from yuccalib.image_processing.transforms.SimulateLowres import SimulateLowres
from yuccalib.image_processing.transforms.Spatial import Spatial
from yuccalib.network_architectures.utils.model_memory_estimation import (
    find_optimal_tensor_dims,
)


class YuccaAugmentationComposer:
    def __init__(
        self,
        patch_size: list | tuple,
        is_2D: bool = False,
        parameter_dict: dict = {},
    ):
        self._pre_aug_patch_size = None

        self.setup_default_params(is_2D, patch_size)
        self.overwrite_params(parameter_dict)
        self.train_transforms = self.compose_train_transforms()
        self.val_transforms = self.compose_val_transforms()

    def setup_default_params(self, is_2d, patch_size):
        print("Composing Transforms")
        # Define whether we crop before or after applying augmentations
        # Define if cropping is random or always centered
        self.random_crop = True
        self.mask_image_for_reconstruction = False
        self.patch_size = patch_size

        # label/segmentation transforms
        self.skip_seg = False
        self.seg_dtype = int
        self.copy_image_to_seg = False

        self.additive_noise_p_per_sample = 0.2
        self.additive_noise_mean = (0.0, 0.0)
        self.additive_noise_sigma = (1e-3, 1e-4)

        self.biasfield_p_per_sample = 0.33

        self.blurring_p_per_sample = 0.2
        self.blurring_sigma = (0.0, 1.0)
        self.blurring_p_per_channel = 0.5

        self.elastic_deform_p_per_sample = 0.33
        self.elastic_deform_alpha = (200, 600)
        self.elastic_deform_sigma = (20, 30)

        self.gamma_p_per_sample = 0.2
        self.gamma_p_invert_image = 0.05
        self.gamma_range = (0.5, 2.0)

        self.gibbs_ringing_p_per_sample = 0.2
        self.gibbs_ringing_cutfreq = (96, 129)
        self.gibbs_ringing_axes = (0, 2) if is_2d else (0, 3)

        self.mirror_p_per_sample = 0.0
        self.mirror_p_per_axis = 0.33
        self.mirror_axes = (0, 1) if is_2d else (0, 1, 2)

        self.motion_ghosting_p_per_sample = 0.2
        self.motion_ghosting_alpha = (0.85, 0.95)
        self.motion_ghosting_numreps = (2, 11)
        self.motion_ghosting_axes = (0, 2) if is_2d else (0, 3)

        self.multiplicative_noise_p_per_sample = 0.2
        self.multiplicative_noise_mean = (0, 0)
        self.multiplicative_noise_sigma = (1e-3, 1e-4)

        self.rotation_p_per_sample = 0.2
        self.rotation_p_per_axis = 0.66
        self.rotation_x = (-30.0, 30.0)
        self.rotation_y = (-0.0, 0.0) if is_2d else (-30.0, 30.0)
        self.rotation_z = (-0.0, 0.0) if is_2d else (-30.0, 30.0)

        self.scale_p_per_sample = 0.2
        self.scale_factor = (0.9, 1.1)

        self.simulate_lowres_p_per_sample = 0.2
        self.simulate_lowres_p_per_channel = 0.5
        self.simulate_lowres_p_per_axis = 0.33
        self.simulate_lowres_zoom_range = (0.5, 1.0)

    @property
    def pre_aug_patch_size(self):
        # First check if any spatial transforms are included
        if self.elastic_deform_p_per_sample > 0 or self.rotation_p_per_sample > 0 or self.scale_p_per_sample > 0:
            self._pre_aug_patch_size = get_max_rotated_size(self.patch_size)
        return self._pre_aug_patch_size

    def overwrite_params(self, parameter_dict):
        for key, value in parameter_dict.items():
            self.key = value

    def compose_train_transforms(self):
        tr_transforms = transforms.Compose(
            [
                AddBatchDimension(),
                Spatial(
                    patch_size=self.patch_size,
                    crop=True,
                    random_crop=self.random_crop,
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
                    skip_seg=self.skip_seg,
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
                    skip_seg=self.skip_seg,
                ),
                # DownsampleSegForDS() if self.deep_supervision else None,
                # CopyImageToSeg() if self.CopyImageToSeg else None,
                # Masking() if self.MaskImageForReconstruction else None,
                RemoveBatchDimension(),
            ]
        )
        return tr_transforms

    def compose_val_transforms(self):
        val_transforms = transforms.Compose(
            [
                # AddBatchDimension(),
                # CopyImageToSeg() if self.CopyImageToSeg else None,
                # Masking() if self.MaskImageForReconstruction else None,
                # RemoveBatchDimension(),
            ]
        )
        return val_transforms


if __name__ == "__main__":
    x = YuccaAugmentationComposer()
