from torchvision import transforms
from yuccalib.image_processing.transforms.formatting import AddBatchDimension, RemoveBatchDimension
from yuccalib.image_processing.transforms.BiasField import BiasField
from yuccalib.image_processing.transforms.Blur import Blur
from yuccalib.image_processing.transforms.CopyImageToSeg import CopyImageToSeg
from yuccalib.image_processing.transforms.Gamma import Gamma
from yuccalib.image_processing.transforms.Ghosting import MotionGhosting
from yuccalib.image_processing.transforms.Masking import Masking
from yuccalib.image_processing.transforms.Mirror import Mirror
from yuccalib.image_processing.transforms.Noise import AdditiveNoise, MultiplicativeNoise
from yuccalib.image_processing.transforms.Ringing import GibbsRinging
from yuccalib.image_processing.transforms.sampling import DownsampleSegForDS
from yuccalib.image_processing.transforms.SimulateLowres import SimulateLowres
from yuccalib.image_processing.transforms.Spatial import Spatial
from yuccalib.network_architectures.utils.model_memory_estimation import find_optimal_tensor_dims



class YuccaAugmenter:
	def __init__(self,
			  patch_size: list | tuple = None,
			  is_2D: bool = False,
			  parameter_dict: dict = {}):
		self.setup_default_params(is_2D, patch_size)
		self.overwrite_params(parameter_dict)
		self.tr_transforms = self.compose_tr_transforms()
		self.val_transforms = self.compose_val_transforms()

	def setup_default_params(self, is_2D, patch_size):
		# Define whether we crop before or after applying augmentations
		# Define if cropping is random or always centered
		self.RandomCrop = True
		self.MaskImageForReconstruction = False
		self.patch_size = patch_size

		# Label/segmentation transforms
		self.SkipSeg = False
		self.SegDtype = int
		self.CopyImageToSeg = False

		self.AdditiveNoise_p_per_sample = 0.2
		self.AdditiveNoise_mean = (0., 0.)
		self.AdditiveNoise_sigma = (1e-3, 1e-4)

		self.BiasField_p_per_sample = 0.33

		self.Blurring_p_per_sample = 0.2
		self.Blurring_sigma = (0., 1.)
		self.Blurring_p_per_channel = 0.5

		self.ElasticDeform_p_per_sample = 0.33
		self.ElasticDeform_alpha = (200, 600)  
		self.ElasticDeform_sigma = (20, 30)

		self.Gamma_p_per_sample = 0.2
		self.Gamma_p_invert_image = 0.05
		self.Gamma_range = (0.5, 2.)

		self.GibbsRinging_p_per_sample = 0.2
		self.GibbsRinging_cutFreq = (96, 129)
		self.GibbsRinging_axes = (0, 3)

		self.Mirror_p_per_sample = 0.0
		self.Mirror_p_per_axis = 0.33
		self.Mirror_axes = (0, 1, 2)

		self.MotionGhosting_p_per_sample = 0.2
		self.MotionGhosting_alpha = (0.85, 0.95)
		self.MotionGhosting_numReps = (2, 11)
		self.MotionGhosting_axes = (0, 3)

		self.MultiplicativeNoise_p_per_sample = 0.2
		self.MultiplicativeNoise_mean = (0, 0)
		self.MultiplicativeNoise_sigma = (1e-3, 1e-4)

		self.Rotation_p_per_sample = 0.2
		self.Rotation_p_per_axis = 0.66
		self.Rotation_x = (-30., 30.)
		self.Rotation_y = (-30., 30.)
		self.Rotation_z = (-30., 30.)

		self.Scale_p_per_sample = 0.2
		self.Scale_factor = (0.9, 1.1)

		self.SimulateLowres_p_per_sample = 0.2
		self.SimulateLowres_p_per_channel = 0.5
		self.SimulateLowres_p_per_axis = 0.33
		self.SimulateLowres_zoom_range = (0.5, 1.)

		if is_2D:
			self.GibbsRinging_axes = (0, 2)
			self.Mirror_axes = (0, 1)
			self.MotionGhosting_axes = (0, 2)
			self.Rotation_y = (-0., 0.)
			self.Rotation_z = (-0., 0.)

	def overwrite_params(self, parameter_dict):
		for key, value in parameter_dict.items():
			self.key = value

	def compose_tr_transforms(self):
		tr_transforms = transforms.Compose([
			AddBatchDimension(),
			Spatial(
				patch_size=self.patch_size, 
				crop=True,
				random_crop=self.RandomCrop,
				p_deform_per_sample=self.ElasticDeform_p_per_sample,
				deform_sigma=self.ElasticDeform_sigma,
				deform_alpha=self.ElasticDeform_alpha,
				p_rot_per_sample=self.Rotation_p_per_sample,
				p_rot_per_axis=self.Rotation_p_per_axis,
				x_rot_in_degrees=self.Rotation_x,
				y_rot_in_degrees=self.Rotation_y,
				z_rot_in_degrees=self.Rotation_z,
				p_scale_per_sample=self.Scale_p_per_sample,
				scale_factor=self.Scale_factor,
				skip_seg=self.SkipSeg),
			AdditiveNoise(
				p_per_sample=self.AdditiveNoise_p_per_sample,
				mean=self.AdditiveNoise_mean,
				sigma=self.AdditiveNoise_sigma),
			Blur(
				p_per_sample=self.Blurring_p_per_sample,
				p_per_channel=self.Blurring_p_per_channel,
				sigma=self.Blurring_sigma),
			MultiplicativeNoise(
				p_per_sample=self.MultiplicativeNoise_p_per_sample,
				mean=self.MultiplicativeNoise_mean,
				sigma=self.MultiplicativeNoise_sigma),
			MotionGhosting(
				p_per_sample=self.MotionGhosting_p_per_sample,
				alpha=self.MotionGhosting_alpha,
				numReps=self.MotionGhosting_numReps,
				axes=self.MotionGhosting_axes),
			GibbsRinging(
				p_per_sample=self.GibbsRinging_p_per_sample,
				cutFreq=self.GibbsRinging_cutFreq,
				axes=self.GibbsRinging_axes),
			SimulateLowres(
				p_per_sample=self.SimulateLowres_p_per_sample,
				p_per_channel=self.SimulateLowres_p_per_channel,
				p_per_axis=self.SimulateLowres_p_per_axis,
				zoom_range=self.SimulateLowres_zoom_range),
			BiasField(
				p_per_sample=self.BiasField_p_per_sample),
			Gamma(
				p_per_sample=self.Gamma_p_per_sample,
				p_invert_image=self.Gamma_p_invert_image,
				gamma_range=self.Gamma_range),
			Mirror(
				p_per_sample=self.Mirror_p_per_sample,
				axes=self.Mirror_axes,
				p_mirror_per_axis=self.Mirror_p_per_axis,
				skip_seg=self.SkipSeg),
			#DownsampleSegForDS() if self.deep_supervision else None,
			#CopyImageToSeg() if self.CopyImageToSeg else None,
			#Masking() if self.MaskImageForReconstruction else None,
			RemoveBatchDimension()])
		return tr_transforms
	
	def compose_val_transforms(self):		
		val_transforms = transforms.Compose([
			AddBatchDimension(),
			#CopyImageToSeg() if self.CopyImageToSeg else None,
			#Masking() if self.MaskImageForReconstruction else None,
			RemoveBatchDimension()])
		return val_transforms