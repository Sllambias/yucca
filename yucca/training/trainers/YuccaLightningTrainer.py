#%%
import lightning as pl
import torch
from yuccalib.utils.files_and_folders import recursive_find_python_class
from yuccalib.loss_and_optim.loss_functions.CE import CE
import yuccalib


class YuccaLightningModule(pl.LightningModule):
    def __init__(self, model_name: str = 'UNet'):
        super().__init__()
        self.model_name = model_name
        #self.loss_fn = 

        self.save_hyperparameters()
        self.initialize()


    def initialize(self):
        self.model = recursive_find_python_class(folder=[join(yuccalib.__path__[0], 'network_architectures')],
                                            class_name=self.model_name,
                                            current_module='yuccalib.network_architectures')

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)
    

#%%

from yucca.yucca.training.data_loading.YuccaLightningLoader import YuccaDataModule
dm = YuccaDataModule(r'/Users/zcr545/Desktop/Projects/YuccaData/yucca_preprocessed/Task001_OASIS/YuccaPlanner')
model = YuccaLightningModule()

dm.setup("1")
tdl = dm.train_dataloader()
model.fit(tdl)
#%%
i = 0
while i < 10:
	for out in tdl:
		x = model(out)
		print(x['image'].shape)
		i += 1
		#print(out)

# %%
