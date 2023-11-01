#%%
import lightning as pl
import torch
import yuccalib
from batchgenerators.utilities.file_and_folder_operations import join
from yuccalib.utils.files_and_folders import recursive_find_python_class
from yuccalib.loss_and_optim.loss_functions.CE import CE
from yuccalib.image_processing.transforms.BiasField import BiasField
from yuccalib.image_processing.transforms.Spatial import Spatial
from yuccalib.image_processing.transforms.formatting import NumpyToTorch
from yuccalib.image_processing.transforms.cropping_and_padding import CropPad
from torchvision import transforms


class YuccaLightningModule(pl.LightningModule):
	def __init__(self, model_name: str = 'UNet'):
		super().__init__()
		# Model parameters
		self.model_name = model_name

		# Loss, optimizer and scheduler parameters
		self.loss_fn = CE


		self.save_hyperparameters()
		self.initialize()


	def initialize(self):
		self.model = recursive_find_python_class(folder=[join(yuccalib.__path__[0], 'network_architectures')],
											class_name=self.model_name,
											current_module='yuccalib.network_architectures')
		self.model = self.model(input_channels=1, conv_op=torch.nn.Conv3d, norm_op=torch.nn.InstanceNorm3d, num_classes=2)

	def forward(self, inputs):
		return self.model(inputs)


	def training_step(self, batch, batch_idx):
		inputs, target = batch['image'], batch['seg']
		output = self(inputs.float())
		loss = self.loss_fn(output.softmax(1), target)
		return loss

	def configure_optimizers(self):
		self.loss_fn = self.loss_fn()
		return torch.optim.SGD(self.model.parameters(), lr=0.1)
	

if __name__ == '__main__':
	from yucca.training.data_loading.YuccaLightningLoader import YuccaDataModule
	dm = YuccaDataModule(r'/Users/zcr545/Desktop/Projects/YuccaData/yucca_preprocessed/Task001_OASIS/YuccaPlanner', batch_size=2)
	model = YuccaLightningModule()

	dm.setup("1")
	tdl = dm.train_dataloader()
	trainer = pl.Trainer(fast_dev_run=2, max_epochs=1)
	trainer.fit(model=model, train_dataloaders=tdl)
	#i = 0
	#while i < 2:
	#	for x in tdl:
	#		i += 1
	#		print(x['image'].shape)

# %%
