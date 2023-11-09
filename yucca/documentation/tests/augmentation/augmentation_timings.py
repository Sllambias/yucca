from yucca.image_processing.transforms.Spatial import Spatial
from yucca.image_processing.transforms.BiasField import BiasField
from yucca.image_processing.transforms.Noise import AdditiveNoise, MultiplicativeNoise
from yucca.image_processing.transforms.Ghosting import MotionGhosting
from yucca.image_processing.transforms.Ringing import GibbsRinging
from nibabel.testing import data_path
import nibabel as nib
import os
from copy import deepcopy
import numpy as np
import timeit

np.random.seed(4215235)

# We use a publicly available sample from Nibabel
example_file = os.path.join(data_path, "example4d.nii.gz")
im = nib.load(example_file)
imarr = im.get_fdata()[:, :, :, 0]

# Convert it to the format expected of the transforms
# a dict of {"image": image, "seg": segmentation}
# with samples for dims (b, c, x, y, z) for 3D or (b, c, x, y) for 2D
imarr = imarr[np.newaxis, np.newaxis]
seg = np.zeros(imarr.shape)
datadict = {"image": imarr, "seg": seg}

# Applying Rotation
tform = Spatial(do_rot=True, x_rot_in_degrees=(0, 0), y_rot_in_degrees=(0, 0), z_rot_in_degrees=(45, 45))
print("Rotation: ", timeit.timeit(lambda: tform(**deepcopy(datadict)), number=100))

# Applying Deformation
tform = Spatial(do_deform=True, deform_alpha=(700, 700), deform_sigma=(10, 10))
print("Deformation: ", timeit.timeit(lambda: tform(**deepcopy(datadict)), number=100))

# Applying Scaling
tform = Spatial(do_scale=True, scale_factor=(1.5, 1.5))
print("Scaling: ", timeit.timeit(lambda: tform(**deepcopy(datadict)), number=100))

# Applying Cropping
tform = Spatial(do_crop=True, patch_size=np.array(imarr.shape[2:]) // 2)
print("Scaling: ", timeit.timeit(lambda: tform(**deepcopy(datadict)), number=100))

# Applying Rotation, Deformation and Scaling WITHOUT cropping
tform = Spatial(
    do_rot=True,
    x_rot_in_degrees=(0, 0),
    y_rot_in_degrees=(0, 0),
    z_rot_in_degrees=(45, 45),
    do_deform=True,
    deform_alpha=(700, 700),
    deform_sigma=(10, 10),
    do_scale=True,
    scale_factor=(1.5, 1.5),
)
print("Rotation, Deformation and Scaling WITHOUT cropping: ", timeit.timeit(lambda: tform(**deepcopy(datadict)), number=100))

# Applying Rotation, Deformation and Scaling WITH cropping
tform = Spatial(
    do_crop=True,
    patch_size=np.array(imarr.shape[2:]) // 2,
    do_rot=True,
    x_rot_in_degrees=(0, 0),
    y_rot_in_degrees=(0, 0),
    z_rot_in_degrees=(45, 45),
    do_deform=True,
    deform_alpha=(700, 700),
    deform_sigma=(10, 10),
    do_scale=True,
    scale_factor=(1.5, 1.5),
)
print("Rotation, Deformation and Scaling WITH cropping: ", timeit.timeit(lambda: tform(**deepcopy(datadict)), number=100))

# Applying Additive Noise
tform = AdditiveNoise(p_per_sample=1, mean=(0, 0), sigma=(25, 25))
print("Additive Noise: ", timeit.timeit(lambda: tform(**deepcopy(datadict)), number=100))

# Applying Multiplicative Noise
tform = MultiplicativeNoise(p_per_sample=1, mean=(0, 0), sigma=(0.1, 0.1))
print("Multiplicative Noise: ", timeit.timeit(lambda: tform(**deepcopy(datadict)), number=100))

# Applying Bias Field
tform = BiasField(p_per_sample=1)
print("Bias Fields: ", timeit.timeit(lambda: tform(**deepcopy(datadict)), number=100))

# Applying Motion Ghosting
tform = MotionGhosting(p_per_sample=1, alpha=(0.6, 0.6), numReps=(3, 4), dims=(0, 1))
print("Motion Ghosting: ", timeit.timeit(lambda: tform(**deepcopy(datadict)), number=100))

# Applying Gibbs Ringing
tform = GibbsRinging(p_per_sample=1, cutFreq=(30, 36), dim=(0, 1))
print("Gibbs Ringing: ", timeit.timeit(lambda: tform(**deepcopy(datadict)), number=100))
