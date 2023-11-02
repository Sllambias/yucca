#%%
import lightning as pl
from batchgenerators.utilities.file_and_folder_operations import join
from yucca.training.augmentation.YuccaAugmenter import YuccaAugmenter
from yucca.training.data_loading.YuccaDataModule import YuccaDataModule
from yucca.training.trainers.YuccaConfigurator import YuccaConfigurator
from yucca.training.trainers.YuccaLightningModule import YuccaLightningModule


class YuccaLightningTrainer(pl.Trainer):
	"""
	The YuccaLightningWrapper is the one to rule them all.
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
		super().__init__(**kwargs)
		
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
		self.is_initialized = False

	def initialize(self):
		if not self.is_initialized:
			# Here we configure the outpath we will use to store model files and metadata
			# along with the path to plans file which will also be loaded.
			configurator = YuccaConfigurator(
				folds = self.folds,
				model_dimensions=self.model_dimensions,
				model_name=self.model_name,
				planner=self.planner,
				task=self.task)

			# Based on the plans file loaded above, this will load information about the expected
			# number of modalities and classes in the dataset and compute the optimal batch and
			# patch sizes given the specified network architecture.
			augmenter = YuccaAugmenter(is_2D=False)

			self.data_module = YuccaDataModule(
				preprocessed_data_dir=join(yucca_preprocessed, self.task, self.planner), 
				batch_size=configurator.batch_size,
				composed_tr_transforms=augmenter.tr_transforms,
				composed_val_transforms=augmenter.val_transforms,
				generator_patch_size=configurator.initial_patch_size)
			
			self.model_module = YuccaLightningModule(
				num_classes = configurator.num_classes,
				num_modalities = configurator.num_modalities)

			if self.continue_training:
				# Do something here - find last model ckpt and replace self.continue_training
				# with the path to this
				pass

			self.is_initialized = True
		else:
			print("Network is already initialized. \
				  Calling initialize repeatedly should be avoided.")
	
	def run_training(self):
		self.initialize()
		super().fit(self.model_module, self.data_module)


if __name__ == '__main__':
	from batchgenerators.utilities.file_and_folder_operations import join

	trainer = YuccaLightningTrainer(task = 'Task001_OASIS', fast_dev_run=2, max_epochs=1, default_root_dir=None)
	#trainer.fit()
#%%
