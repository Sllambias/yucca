#%%
import lightning as pl
import torch
import yuccalib
import torch.nn as nn
from torchvision import transforms
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p
from yucca.paths import yucca_models, yucca_preprocessed
from yuccalib.utils.files_and_folders import recursive_find_python_class
from yuccalib.loss_and_optim.loss_functions.CE import CE
from yuccalib.utils.kwargs import filter_kwargs
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
from yuccalib.image_processing.matrix_ops import get_max_rotated_size

class YuccaLightningTrainer:
	"""
	The YuccaLightningTrainer is the one to rule them all.
	This will take ALL the arguments you need for training and apply them accordingly.

	First it automatically configure a host of parameters for you - such as batch size, 
	patch size, modalities, classes, augmentations (and their hyperparameters) and formatting.
	It also instantiates the PyTorch Lightning trainer.
	
	Then, it will instantiate the YuccaLightningModule - containing the model, optimizers, 
	loss functions and learning rates.

	Then, it will call the YuccaLightningDataModule. The DataModule creates the dataloaders that 
	we use iterate over the chosen Task, which is wrapped in a YuccaDataset.

	YuccaLightningTrainer
	├── model_params
	├── data_params
	├── aug_params
	├── pl.Trainer
	|
	├── YuccaLightningModule(model_params) -> model
	|   ├── network(model_params)
	|   ├── optim
	|   ├── loss_fn
	|   ├── scheduler
	|
	├── YuccaLightningDataModule(data_params, aug_params) -> train_dataloader, val_dataloader
	|   ├── YuccaDataset(data_params, aug_params)
	|   ├── InfiniteRandomSampler
	|   ├── DataLoaders(YuccaDataset, InfiniteRandomSampler)
	|
	├── pl.Trainer.fit(model, train_dataloader, val_dataloader)
	| 
	"""
	def __init__(
			self, 
			continue_training: bool = None,
			deep_supervision: bool = False,
			folds: int = 0,
			model_dimensions: str = '3D',
			model_name: str = 'UNet',
			planner: str = 'YuccaPlanner',
			task: str = None,
			**kwargs
			):
		
		self.continue_training = continue_training
		self.deep_supervision = deep_supervision
		self.folds = folds
		self.model_dimensions = model_dimensions
		self.model_name = model_name
		self.name = self.__class__.__name__
		self.planner = planner
		self.task = task

		# default settings
		self.max_vram = 2
		self.trainer = pl.Trainer(**kwargs)

		# Start
		self.initialize()

	def configure_paths(self):
		self.outpath = join(
			yucca_models, 
			self.task,
			self.model_name + '__' + self.planner, 
			self.model_dimensions,
			self.name, str(self.folds))
		
		maybe_mkdir_p(self.outpath)

		self.plans_path = join(
			yucca_preprocessed, 
			self.task,
			self.planner,
			self.planner + '_plans.json')
		
		self.plans = load_json(self.plans_path)

	def configure_train_params(self):
		self.num_classes = len(self.plans['dataset_properties']['classes'])
		self.num_modalities = len(self.plans['dataset_properties']['modalities'])
		if torch.cuda.is_available():
			self.batch_size, self.patch_size = find_optimal_tensor_dims(dimensionality=self.model_dimensions,
																		num_classes=self.num_classes,
																		modalities=self.num_modalities,
																		model_name=self.model_name,
																		max_patch_size=self.plans['new_mean_size'],
																		max_memory_usage_in_gb=self.max_vram)
		else:
			print("Cuda is not available, using tiny patch and batch")
			self.batch_size = 2
			self.patch_size = (32, 32, 32)
		self.initial_patch_size = get_max_rotated_size(self.patch_size)

	def initialize(self):
		self.configure_paths()
		self.plans = load_json(self.plans_path)
		self.configure_train_params()
		self.model = self.initialize_lightningmodule()
		self.dm = self.initialize_datamodule()
		if self.continue_training:
			# Do something here - find last model ckpt and replace self.continue_training
			# with the path to this
			pass

	def initialize_lightningmodule(self):
		return YuccaLightningModule(
			num_classes = self.num_classes,
			num_modalities = self.num_modalities)

	def initialize_datamodule(self):		
		composed_tr_transforms, composed_val_transforms = self.get_composed_transforms()
		return YuccaDataModule(
			preprocessed_data_dir=join(yucca_preprocessed, 'Task001_OASIS/YuccaPlanner'), 
			batch_size=2,
			composed_tr_transforms=composed_tr_transforms,
			composed_val_transforms=composed_val_transforms,
			generator_patch_size=self.initial_patch_size)

	def get_train_generators(self):
		self.dm.setup("1")
		self.tr_dl = self.dm.train_dataloader()
		self.val_dl = self.dm.val_dataloader()
		return self.tr_dl, self.val_dl
	
	def fit(self, **kwargs):
		self.trainer.fit(**kwargs)
		
	def setup_DA(self):
		# Define whether we crop before or after applying augmentations
		# Define if cropping is random or always centered
		self.RandomCrop = True
		self.MaskImageForReconstruction = False

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

		if self.model_dimensions == '2D':
			self.GibbsRinging_axes = (0, 2)
			self.Mirror_axes = (0, 1)
			self.MotionGhosting_axes = (0, 2)
			self.Rotation_y = (-0., 0.)
			self.Rotation_z = (-0., 0.)
	

	def get_composed_transforms(self):
		self.setup_DA()
		composed_tr_transforms = transforms.Compose([
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

		composed_val_transforms = transforms.Compose([
			AddBatchDimension(),
			#CopyImageToSeg() if self.CopyImageToSeg else None,
			#Masking() if self.MaskImageForReconstruction else None,
			RemoveBatchDimension()])
		
		return composed_tr_transforms, composed_val_transforms

class YuccaLightningModule(pl.LightningModule):
	def __init__(
			self,
			learning_rate: float = 1e-3,
			loss_fn: nn.Module = CE,
			model_name: str = 'UNet',
			model_dimensions: str = '3D',
			momentum: float = 0.9,
			num_modalities: int = 1,
			num_classes: int = 1,
			optimizer: torch.optim.Optimizer = torch.optim.SGD,
			):
		super().__init__()
		# Model parameters
		self.model_name = model_name
		self.model_dimensions = model_dimensions
		self.num_classes = num_classes
		self.num_modalities = num_modalities

		# Loss, optimizer and scheduler parameters
		self.lr = learning_rate
		self.loss_fn = loss_fn
		self.momentum = momentum
		self.optim = optimizer

		# Save params and start training
		self.save_hyperparameters()
		self.initialize()


	def initialize(self):
		self.model = recursive_find_python_class(folder=[join(yuccalib.__path__[0], 'network_architectures')],
											class_name=self.model_name,
											current_module='yuccalib.network_architectures')
		
		if self.model_dimensions == '3D':
			conv_op = torch.nn.Conv3d
			norm_op = torch.nn.InstanceNorm3d

		self.model = self.model(
			input_channels=self.num_modalities, 
			conv_op=conv_op, 
			norm_op=norm_op, 
			num_classes=self.num_classes)

	def forward(self, inputs):
		return self.model(inputs)

	def training_step(self, batch, batch_idx):
		inputs, target = batch['image'], batch['seg']
		output = self(inputs.float())
		print(output.size(), target.size())
		loss = self.loss_fn(output.softmax(1), target)
		print(loss)
		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return loss

	def configure_optimizers(self):
		# Initialize the loss function using relevant kwargs
		loss_kwargs = {}
		loss_kwargs = filter_kwargs(self.loss_fn, loss_kwargs)
		self.loss_fn = self.loss_fn(**loss_kwargs)

		# optim_kwargs holds arguments for all possible optimizers we could use.
		# The subsequent filter_kwargs will select only the kwargs that can be passed 
		# to the chosen optimizer
		optim_kwargs = {
			'lr': self.lr,
			'momentum': self.momentum}
		optim_kwargs = filter_kwargs(self.optim, optim_kwargs)
		return self.optim(self.model.parameters(), lr=self.lr)
	

if __name__ == '__main__':
	from yucca.training.data_loading.YuccaLightningLoader import YuccaDataModule
	from yucca.paths import yucca_preprocessed
	from batchgenerators.utilities.file_and_folder_operations import join

	trainer = YuccaLightningTrainer(task = 'Task001_OASIS', fast_dev_run=2, max_epochs=1, default_root_dir=None)
	tdl, vdl = trainer.get_train_generators()
	trainer.fit(model=trainer.model, train_dataloaders=tdl, val_dataloaders=vdl, ckpt_path=None)


# %%
