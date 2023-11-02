#%%
import yucca
import lightning as L
import numpy as np
import os
from batchgenerators.utilities.file_and_folder_operations import join, subfiles
from yucca.training.augmentation.YuccaAugmentationComposer import YuccaAugmentationComposer
from yucca.training.data_loading.YuccaDataModule import YuccaDataModule
from yucca.training.trainers.YuccaConfigurator import YuccaConfigurator
from yucca.training.trainers.YuccaLightningModule import YuccaLightningModule
from yucca.preprocessing.YuccaPreprocessor import YuccaPreprocessor
from yucca.paths import yucca_preprocessed
from yuccalib.utils.files_and_folders import recursive_find_python_class

class YuccaLightningManager:
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
            ckpt_path: str = None,
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
        self.ckpt_path = ckpt_path

        # default settings
        self.max_vram = 2
        self.is_initialized = False
        self.kwargs = kwargs
        self.trainer = L.Trainer

    def initialize(self, train=False, pred_data_dir: str = None):
            # Here we configure the outpath we will use to store model files and metadata
            # along with the path to plans file which will also be loaded.
        configurator = YuccaConfigurator(
            folds = self.folds,
            model_dimensions=self.model_dimensions,
            model_name=self.model_name,
            planner=self.planner,
            task=self.task)

        self.model_module = YuccaLightningModule(
            num_classes = configurator.num_classes,
            num_modalities = configurator.num_modalities)

        #if self.ckpt_path:
        #    self.model_module = YuccaLightningModule.load_from_checkpoint(self.ckpt_path)
            

        augmenter = YuccaAugmentationComposer(
            patch_size=configurator.patch_size,
            is_2D=False)

        self.data_module = YuccaDataModule(
            train_data_dir=configurator.train_data_dir,
            pred_data_dir=pred_data_dir,
            batch_size=configurator.batch_size,
            composed_train_transforms=augmenter.train_transforms,
            composed_val_transforms=augmenter.val_transforms,
            generator_patch_size=configurator.initial_patch_size)
        
        self.trainer = L.Trainer(default_root_dir=configurator.outpath, **self.kwargs)

        if not train:
            self.preprocessor = YuccaPreprocessor(configurator.plans_path)

    def run_training(self):
        self.initialize(train=True)
        self.trainer.fit(model = self.model_module, 
                         datamodule=self.data_module, 
                         ckpt_path=self.ckpt_path)

    def predict_folder(self, input_folder, output_folder, not_strict=True, save_softmax=False,
                       overwrite=False, do_tta=False):
        self.initialize(train=False, pred_data_dir = input_folder)
        print(self.model_module)
        self.trainer.predict(model = self.model_module, 
                             dataloaders=self.data_module, 
                             ckpt_path=self.ckpt_path)
        #
        #files = subfiles(input_folder, suffix='.nii.gz', join=False)
#
        #if not not_strict:
        #    # If strict we enforce modality encoding.
        #    # This means files must be encoded as the model expects.
        #    # e.g. "_000" or "_001" etc., for T1 and T2 scans.
        #    # This allows us to handle different modalities
        #    # of the same subject, rather than treating them as individual cases.
        #    expected_modalities = self.plans['dataset_properties']['modalities']
        #    subject_ids, counts = np.unique([i[:-len('_000.nii.gz')] for i in files], return_counts=True)
        #    assert all(counts == len(expected_modalities)), "Aborting. Modalities are missing for some samples"
        #else:
        #    subject_ids = np.unique([i[:-len('.nii.gz')] for i in files])
#
        #all_cases = []
        #all_outpaths = []
        #for subject_id in subject_ids:
        #    case = [impath for impath in subfiles(input_folder, suffix='.nii.gz') if os.path.split(impath)[-1][:-len('_000.nii.gz')] == subject_id]
        #    outpath = join(output_folder, subject_id)
        #    all_cases.append([self.model_module, case, outpath, save_softmax, overwrite, do_tta])
        #    all_outpaths.append(outpath)
#
        #n_already_predicted = len(subfiles(output_folder, suffix='.nii.gz'))
#
        #print(f"\n"
        #      f"STARTING PREDICTION \n"\
        #      f"{'Cases already predicted: ':25} {n_already_predicted} \n"\
        #      f"{'Cases NOT predicted: ':25} {len(all_outpaths) - n_already_predicted} \n"\
        #      f"{'Overwrite predictions: ':25} {overwrite} \n")
        #print(self.trainer.model)
        #for case_info in all_cases:
        #    self.trainer.predict(*case_info)



#if __name__ == '__main__':
path ='/Users/zcr545/Desktop/Projects/repos/yucca_data/models/Task001_OASIS/UNet__YuccaPlanner/3D/YuccaConfigurator/0/lightning_logs/version_0/checkpoints/epoch=0-step=1.ckpt'
Manager = YuccaLightningManager(task = 'Task001_OASIS', max_epochs=1, ckpt_path=path)
#trainer.run_training()
#trainer.load_checkpoint()
Manager.predict_folder('/Users/zcr545/Desktop/Projects/repos/yucca_data/raw_data/Task001_OASIS/imagesTs',
                       '/Users/zcr545/Desktop/Projects/repos/')
#%%
