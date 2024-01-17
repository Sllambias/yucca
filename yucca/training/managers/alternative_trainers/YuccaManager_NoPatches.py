import lightning as L
import torch
from typing import Literal, Union, Optional
from yucca.training.augmentation.YuccaAugmentationComposer import YuccaAugmentationComposer
from yucca.training.configuration.split_data import get_split_config
from yucca.training.configuration.configure_task import get_task_config
from yucca.training.configuration.configure_callbacks import get_callback_config
from yucca.training.configuration.configure_checkpoint import get_checkpoint_config
from yucca.training.configuration.configure_seed import seed_everything_and_get_seed_config
from yucca.training.configuration.configure_paths import get_path_config
from yucca.training.configuration.configure_plans import get_plan_config
from yucca.training.configuration.input_dimensions import get_input_dims_config
from yucca.training.data_loading.YuccaDataModule import YuccaDataModule
from yucca.training.lightning_modules.YuccaLightningModule import YuccaLightningModule
from yucca.paths import yucca_results


class YuccaManager_NoPatches:
    def __init__(
        self,
        patch_based_training: bool = False,
        **kwargs,
    ):
        super().__init__(
            patch_based_training=patch_based_training,
            **kwargs,
        )
