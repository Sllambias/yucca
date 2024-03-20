no_aug = {
    "additive_noise_p_per_sample": 0.0,
    "biasfield_p_per_sample": 0.0,
    "blurring_p_per_sample": 0.0,
    "blurring_p_per_channel": 0.0,
    "elastic_deform_p_per_sample": 0.0,
    "gamma_p_per_sample": 0.0,
    "gamma_p_invert_image": 0.0,
    "gibbs_ringing_p_per_sample": 0.0,
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
}

all_always = {
    "random_crop": True,
    "mask_image_for_reconstruction": True,
    "skip_label": False,
    "label_dtype": int,
    "copy_image_to_label": False,
    "additive_noise_p_per_sample": 1.0,
    "biasfield_p_per_sample": 1.0,
    "blurring_p_per_sample": 1.0,
    "blurring_p_per_channel": 1.0,
    "elastic_deform_p_per_sample": 1.0,
    "gamma_p_per_sample": 1.0,
    "gamma_p_invert_image": 1.0,
    "gibbs_ringing_p_per_sample": 1.0,
    "mirror_p_per_sample": 1.0,
    "mirror_p_per_axis": 1.0,
    "motion_ghosting_p_per_sample": 1.0,
    "multiplicative_noise_p_per_sample": 1.0,
    "rotation_p_per_sample": 1.0,
    "rotation_p_per_axis": 1.0,
    "scale_p_per_sample": 1.0,
    "simulate_lowres_p_per_sample": 1.0,
    "simulate_lowres_p_per_channel": 1.0,
    "simulate_lowres_p_per_axis": 1.0,
}


basic = no_aug.copy()
basic["mirror_p_per_sample"] = 0.2
basic["mirror_p_per_axis"] = 0.66
basic["rotation_p_per_sample"] = 0.2
basic["rotation_p_per_axis"] = 0.66
basic["scale_p_per_sample"] = 0.2
