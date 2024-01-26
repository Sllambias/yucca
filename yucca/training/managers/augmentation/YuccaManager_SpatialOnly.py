from yucca.training.managers.YuccaManager import YuccaManager


class YuccaManager_SpatialOnly(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = {
            "additive_noise_p_per_sample": 0.0,
            "blurring_p_per_sample": 0.0,
            "multiplicative_noise_p_per_sample": 0.0,
            "motion_ghosting_p_per_sample": 0.0,
            "gibbs_ringing_p_per_sample": 0.0,
            "simulate_lowres_p_per_sample": 0.0,
            "biasfield_p_per_sample": 0.0,
            "gamma_p_per_sample": 0.0,
            "mirror_p_per_sample": 0.0,
        }
