from batchgenerators.utilities.file_and_folder_operations import join, isdir, subdirs, maybe_mkdir_p
from dataclasses import dataclass, asdict
from typing import Union, Tuple
from yucca.paths import yucca_models, yucca_preprocessed_data
from yucca.training.configuration.configure_task import TaskConfig


@dataclass
class AugmenterConfig:
    mask_image_for_reconstruction: bool

    # label/segmentation transforms
    skip_label: bool
    label_dtype: type
    copy_image_to_label: bool

    # AdditiveNoise params
    additive_noise_p_per_sample: float
    additive_noise_mean: Tuple[float, float]
    additive_noise_sigma: Tuple[float, float]

    # BiasField params
    biasfield_p_per_sample: float

    # Blur params
    blurring_p_per_sample: float
    blurring_sigma: Tuple[float, float]
    blurring_p_per_channel: float

    # Spatial crop params
    random_crop: bool
    crop: bool

    # Spatial deform params
    elastic_deform_p_per_sample: float
    elastic_deform_alpha: Tuple[int, int]
    elastic_deform_sigma: Tuple[int, int]

    # Gamma params
    gamma_p_per_sample: float
    gamma_p_invert_image: float
    gamma_range: Tuple[float, float]

    # GibbsRinging params
    gibbs_ringing_p_per_sample: float
    gibbs_ringing_cutfreq: Tuple[int, int]
    gibbs_ringing_axes: Tuple[int, int]

    # Mirror params
    mirror_p_per_sample: float
    mirror_p_per_axis: float
    mirror_axes: Union[Tuple[int, int], Tuple[int, int, int]]

    # MotionGhosting params
    motion_ghosting_p_per_sample: float
    motion_ghosting_alpha: Tuple[float, float]
    motion_ghosting_numreps: Tuple[int, int]
    motion_ghosting_axes: Tuple[int, int]

    # MultiplicativeNoise params
    multiplicative_noise_p_per_sample: float
    multiplicative_noise_mean: Tuple[float, float]
    multiplicative_noise_sigma: Tuple[float, float]

    # Spatial rotation params
    rotation_p_per_sample: float
    rotation_p_per_axis: float
    rotation_x: Tuple[float, float]
    rotation_y: Tuple[float, float]
    rotation_z: Tuple[float, float]

    # Spatial scale params
    scale_p_per_sample: float
    scale_factor: Tuple[float, float]

    # SimulateLowres params
    simulate_lowres_p_per_sample: float
    simulate_lowres_p_per_channel: float
    simulate_lowres_p_per_axis: float
    simulate_lowres_zoom_range: Tuple[float, float]

    def lm_hparams(self):
        return {
            "mask_image_for_reconstruction": self.mask_image_for_reconstruction,
            "skip_label": self.skip_label,
            "label_dtype": self.label_dtype,
            "copy_image_to_label": self.copy_image_to_label,
            "AdditiveNoise": 
                {
                    "additive_p_per_sample": self.additive_noise_p_per_sample,
                    "additive_noise_mean": self.additive_noise_mean,
                    "additive_noise_sigma": self.additive_noise_sigma,
                },
            "BiasField":
                {
                    "biasfield_p_per_sample": self.biasfield_p_per_sample,
                },
            "Blur":
                {
                    "blurring_p_per_sample": self.blurring_p_per_sample,
                    "blurring_sigma": self.blurring_sigma,
                    "blurring_p_per_channel": self.blurring_p_per_channel
                },
            "Spatial":
                {
                    "random_crop": self.random_crop,
                    "crop": self.crop,
                    "elastic_deform_p_per_sample": self.elastic_deform_p_per_sample,
                    "elastic_deform_alpha": self.elastic_deform_alpha,
                    "elastic_deform_sigma": self.elastic_deform_sigma,
                    "rotation_p_per_sample": self.rotation_p_per_sample,
                    "rotation_p_per_axis": self.rotation_p_per_axis,
                    "rotation_x": self.rotation_x,
                    "rotation_y": self.rotation_y,
                    "rotation_z": self.rotation_y,
                    "scale_p_per_sample": self.scale_p_per_sample,
                    "scale_factor": self.scale_factor,
                },
            "Gamma": 
                {
                    "gamma_p_per_sample": self.gamma_p_per_sample,
                    "gamma_p_invert_image": self.gamma_p_invert_image,
                    "gamma_range": self.gamma_range,
                },
            "GibbsRinging":
                {
                    "gibbs_ringing_p_per_sample": self.gibbs_ringing_p_per_sample,
                    "gibbs_ringing_cutfreq": self.gibbs_ringing_cutfreq,
                    "gibbs_ringing_axes": self.gibbs_ringing_axes,
                },
            "Mirror":
                {
                    "mirror_p_per_sample": self.mirror_p_per_sample,
                    "mirror_p_per_axis": self.mirror_p_per_axis,
                    "mirror_axes": self.mirror_axes,
                },
            "MotionGhosting":
                {
                    "motion_ghosting_p_per_sample": self.motion_ghosting_p_per_sample,
                    "motion_ghosting_alpha": self.motion_ghosting_alpha,
                    "motion_ghosting_numreps": self.motion_ghosting_numreps,
                    "motion_ghosting_axes": self.motion_ghosting_axes,
                },
            "MultiplicativeNoise":
                {
                    "multiplicative_noise_p_per_sample": self.multiplicative_noise_p_per_sample,
                    "multiplicative_noise_mean": self.multiplicative_noise_mean,
                    "multiplicative_noise_sigma": self.multiplicative_noise_sigma,
                },
            "SimulateLowres":
                {
                    "simulate_lowres_p_per_sample": self.simulate_lowres_p_per_sample,
                    "simulate_lowres_p_per_channel": self.simulate_lowres_p_per_channel,
                    "simulate_lowres_p_per_axis": self.simulate_lowres_p_per_axis,
                    "simulate_lowres_zoom_range": self.simulate_lowres_zoom_range,
                },
        }


def get_train_augmenter_config(overwrite_defaults: dict, is_2D: bool = False):
    
    # Define whether we crop before or after applying augmentations
        # Define if cropping is random or always centered
    default_config = AugmenterConfig(
        mask_image_for_reconstruction = False,

        skip_label = False,
        label_dtype = int,
        copy_image_to_label = False,

        additive_noise_p_per_sample = 0.2,
        additive_noise_mean = (0.0, 0.0),
        additive_noise_sigma = (1e-3, 1e-4),

        biasfield_p_per_sample = 0.33,

        blurring_p_per_sample = 0.2,
        blurring_sigma = (0.0, 1.0),
        blurring_p_per_channel = 0.5,

        random_crop=True,
        crop=True,

        elastic_deform_p_per_sample = 0.33,
        elastic_deform_alpha = (200, 600),
        elastic_deform_sigma = (20, 30),

        gamma_p_per_sample = 0.2,
        gamma_p_invert_image = 0.05,
        gamma_range = (0.5, 2.0),

        gibbs_ringing_p_per_sample = 0.2,
        gibbs_ringing_cutfreq = (96, 129),
        gibbs_ringing_axes = (0, 2) if is_2D else (0, 3),

        mirror_p_per_sample = 0.0,
        mirror_p_per_axis = 0.33,
        mirror_axes = (0, 1) if is_2D else (0, 1, 2),

        motion_ghosting_p_per_sample = 0.2,
        motion_ghosting_alpha = (0.85, 0.95),
        motion_ghosting_numreps = (2, 11),
        motion_ghosting_axes = (0, 2) if is_2D else (0, 3),

        multiplicative_noise_p_per_sample = 0.2,
        multiplicative_noise_mean = (0, 0),
        multiplicative_noise_sigma = (1e-3, 1e-4),

        rotation_p_per_sample = 0.2,
        rotation_p_per_axis = 0.66,
        rotation_x = (-30.0, 30.0),
        rotation_y = (-0.0, 0.0) if is_2D else (-30.0, 30.0),
        rotation_z = (-0.0, 0.0) if is_2D else (-30.0, 30.0),

        scale_p_per_sample = 0.2,
        scale_factor = (0.9, 1.1),

        simulate_lowres_p_per_sample = 0.2,
        simulate_lowres_p_per_channel = 0.5,
        simulate_lowres_p_per_axis = 0.33,
        simulate_lowres_zoom_range = (0.5, 1.0),
        )
    if overwrite_defaults:
        new_config_dict = asdict(default_config)
        new_config_dict.update(overwrite_defaults)
        return AugmenterConfig(**new_config_dict)
    return default_config

    return PathConfig(
        plans_path=plans_path,
        save_dir=save_dir,
        task_dir=task_dir,
        train_data_dir=train_data_dir,
        version_dir=version_dir,
        version=version,
    )


def detect_version(save_dir, continue_from_most_recent) -> Union[None, int]:
    # If the dir doesn't exist we return version 0
    if not isdir(save_dir):
        return 0

    # The dir exists. Check if any previous version exists in dir.
    previous_versions = subdirs(save_dir, join=False)
    # If no previous version exists we return version 0
    if not previous_versions:
        return 0

    # If previous version(s) exists we can either (1) continue from the newest or
    # (2) create the next version
    if previous_versions:
        newest_version = max([int(i.split("_")[-1]) for i in previous_versions])
        if continue_from_most_recent:
            return newest_version
        else:
            return newest_version + 1
