from yucca.pipeline.managers.YuccaManager import YuccaManager


class YuccaManager_Tester(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = {
            "mirror_p_per_sample": 0.0,
            "mirror_p_per_axis": 0.0,
            "motion_ghosting_p_per_sample": 0.0,
            "multiplicative_noise_p_per_sample": 0.0,
            "rotation_p_per_sample": 0.0,
            "rotation_p_per_axis": 0.0,
            "scale_p_per_sample": 0.0,
            "simulate_lowres_p_per_sample": 0.0,
            "simulate_lowres_p_per_channel": 0.0,
            "simulate_lowres_p_per_axis": 0.0,
            "normalize": True,
            "normalization_scheme": "minmax",
        }
