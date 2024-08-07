{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import lightning as L\n",
    "import os\n",
    "from batchgenerators.utilities.file_and_folder_operations import load_json\n",
    "from yucca.pipeline.managers.YuccaManager import YuccaManager\n",
    "from yucca.paths import yucca_raw_data, yucca_preprocessed_data, yucca_models\n",
    "from yucca.pipeline.configuration.configure_task import TaskConfig\n",
    "from yucca.pipeline.configuration.configure_paths import get_path_config\n",
    "from yucca.pipeline.configuration.configure_callbacks import get_callback_config\n",
    "from yucca.pipeline.configuration.split_data import get_split_config\n",
    "from yucca.pipeline.configuration.configure_input_dims import InputDimensionsConfig\n",
    "from yucca.data.augmentation.YuccaAugmentationComposer import YuccaAugmentationComposer\n",
    "from yucca.data.augmentation.augmentation_presets import generic\n",
    "from yucca.lightning_modules.YuccaLightningModule import YuccaLightningModule\n",
    "from yucca.data.data_modules.YuccaDataModule import YuccaDataModule"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Set some variables that we'll need"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "config = {\n",
    "    \"batch_size\": 2,\n",
    "    \"dims\": \"2D\",\n",
    "    \"deep_supervision\": False,\n",
    "    \"experiment\": \"default\",\n",
    "    \"learning_rate\": 1e-3,\n",
    "    \"loss_fn\": \"DiceCE\",\n",
    "    \"model_name\": \"TinyUNet\",\n",
    "    \"momentum\": 0.99,\n",
    "    \"num_classes\": 3,\n",
    "    \"num_modalities\": 1,\n",
    "    \"patch_size\": (32, 32),\n",
    "    \"plans_name\": \"demo\",\n",
    "    \"plans\": None,\n",
    "    \"split_idx\": 0,\n",
    "    \"split_method\": \"kfold\",\n",
    "    \"split_param\": 5,\n",
    "    \"task\": \"Task001_OASIS\",\n",
    "    \"task_type\": \"segmentation\",\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:root:Reusing already computed split file which was split using the kfold method and parameter 5.\n",
      "INFO:root:YuccaLightningModule initialized with the following config: {'batch_size': 2, 'dims': '2D', 'deep_supervision': False, 'experiment': 'default', 'learning_rate': 0.001, 'loss_fn': 'DiceCE', 'model_name': 'TinyUNet', 'momentum': 0.99, 'num_classes': 3, 'num_modalities': 1, 'patch_size': (32, 32), 'plans_name': 'demo', 'plans': None, 'split_idx': 0, 'split_method': 'kfold', 'split_param': 5, 'task': 'Task001_OASIS', 'task_type': 'segmentation', 'continue_from_most_recent': True, 'manager_name': '', 'model_dimensions': '2D', 'patch_based_training': True, 'planner_name': 'demo', 'plans_path': '/Users/zcr545/Desktop/Projects/repos/yucca_data/preprocessed/Task001_OASIS/demo/demo_plans.json', 'save_dir': '/Users/zcr545/Desktop/Projects/repos/yucca_data/models/Task001_OASIS/TinyUNet__2D/__demo/default/kfold_5_fold_0', 'train_data_dir': '/Users/zcr545/Desktop/Projects/repos/yucca_data/preprocessed/Task001_OASIS/demo', 'version_dir': '/Users/zcr545/Desktop/Projects/repos/yucca_data/models/Task001_OASIS/TinyUNet__2D/__demo/default/kfold_5_fold_0/version_0', 'version': 0, 'wandb_id': 'None'}\n",
      "INFO:root:Deep Supervision Enabled: False\n",
      "INFO:root:Loading Model: 2D TinyUNet\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Composing Transforms\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/zcr545/miniconda3/envs/yuccaenv/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\n",
      "  from .autonotebook import tqdm as notebook_tqdm\n",
      "INFO:root:Using 9 workers\n",
      "INFO:root:Using dataset class: <class 'yucca.data.datasets.YuccaDataset.YuccaTrainDataset'> for train/val and <class 'yucca.data.datasets.YuccaDataset.YuccaTestDataset'> for inference\n"
     ]
    }
   ],
   "source": [
    "input_dims_config = InputDimensionsConfig(\n",
    "    batch_size=config.get(\"batch_size\"), patch_size=config.get(\"patch_size\"), num_modalities=config.get(\"num_modalitites\")\n",
    ")\n",
    "task_config = TaskConfig(\n",
    "    task=config.get(\"task\"),\n",
    "    continue_from_most_recent=True,\n",
    "    experiment=config.get(\"experiment\"),\n",
    "    manager_name=\"\",\n",
    "    model_dimensions=config.get(\"dims\"),\n",
    "    model_name=config.get(\"model_name\"),\n",
    "    patch_based_training=True,\n",
    "    planner_name=config.get(\"plans_name\"),\n",
    "    split_idx=config.get(\"split_idx\"),\n",
    "    split_method=config.get(\"split_method\"),\n",
    "    split_param=config.get(\"split_param\"),\n",
    ")\n",
    "\n",
    "path_config = get_path_config(task_config=task_config)\n",
    "\n",
    "split_config = get_split_config(method=task_config.split_method, param=task_config.split_param, path_config=path_config)\n",
    "\n",
    "callback_config = get_callback_config(\n",
    "    save_dir=path_config.save_dir,\n",
    "    version_dir=path_config.version_dir,\n",
    "    experiment=task_config.experiment,\n",
    "    version=path_config.version,\n",
    "    enable_logging=False,\n",
    ")\n",
    "\n",
    "augmenter = YuccaAugmentationComposer(\n",
    "    deep_supervision=config.get(\"deep_supervision\"),\n",
    "    patch_size=input_dims_config.patch_size,\n",
    "    is_2D=True if config.get(\"dims\") == \"2D\" else False,\n",
    "    parameter_dict=generic,\n",
    "    task_type_preset=config.get(\"task_type\"),\n",
    ")\n",
    "\n",
    "\n",
    "model_module = YuccaLightningModule(\n",
    "    config=config | task_config.lm_hparams() | path_config.lm_hparams() | callback_config.lm_hparams(),\n",
    "    deep_supervision=config.get(\"deep_supervision\"),\n",
    "    learning_rate=config.get(\"learning_rate\"),\n",
    "    loss_fn=config.get(\"loss_fn\"),\n",
    "    momentum=config.get(\"momentum\"),\n",
    ")\n",
    "\n",
    "data_module = YuccaDataModule(\n",
    "    composed_train_transforms=augmenter.train_transforms,\n",
    "    composed_val_transforms=augmenter.val_transforms,\n",
    "    input_dims_config=input_dims_config,\n",
    "    train_data_dir=path_config.train_data_dir,\n",
    "    split_idx=task_config.split_idx,\n",
    "    splits_config=split_config,\n",
    "    task_type=config.get(\"task_type\"),\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "GPU available: True (mps), used: False\n",
      "TPU available: False, using: 0 TPU cores\n",
      "IPU available: False, using: 0 IPUs\n",
      "HPU available: False, using: 0 HPUs\n",
      "/Users/zcr545/miniconda3/envs/yuccaenv/lib/python3.11/site-packages/lightning/pytorch/trainer/setup.py:187: GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.\n",
      "/Users/zcr545/miniconda3/envs/yuccaenv/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/checkpoint_connector.py:186: .fit(ckpt_path=\"last\") is set, but there is no last checkpoint available. No checkpoint will be loaded. HINT: Set `ModelCheckpoint(..., save_last=True)`.\n",
      "INFO:root:Setting up data for stage: TrainerFn.FITTING\n",
      "INFO:root:Training on samples: ['/Users/zcr545/Desktop/Projects/repos/yucca_data/preprocessed/Task001_OASIS/demo/OASIS_1000', '/Users/zcr545/Desktop/Projects/repos/yucca_data/preprocessed/Task001_OASIS/demo/OASIS_1001', '/Users/zcr545/Desktop/Projects/repos/yucca_data/preprocessed/Task001_OASIS/demo/OASIS_1002', '/Users/zcr545/Desktop/Projects/repos/yucca_data/preprocessed/Task001_OASIS/demo/OASIS_1008', '/Users/zcr545/Desktop/Projects/repos/yucca_data/preprocessed/Task001_OASIS/demo/OASIS_1009', '/Users/zcr545/Desktop/Projects/repos/yucca_data/preprocessed/Task001_OASIS/demo/OASIS_1010', '/Users/zcr545/Desktop/Projects/repos/yucca_data/preprocessed/Task001_OASIS/demo/OASIS_1011', '/Users/zcr545/Desktop/Projects/repos/yucca_data/preprocessed/Task001_OASIS/demo/OASIS_1012', '/Users/zcr545/Desktop/Projects/repos/yucca_data/preprocessed/Task001_OASIS/demo/OASIS_1013', '/Users/zcr545/Desktop/Projects/repos/yucca_data/preprocessed/Task001_OASIS/demo/OASIS_1014', '/Users/zcr545/Desktop/Projects/repos/yucca_data/preprocessed/Task001_OASIS/demo/OASIS_1015', '/Users/zcr545/Desktop/Projects/repos/yucca_data/preprocessed/Task001_OASIS/demo/OASIS_1036']\n",
      "INFO:root:Validating on samples: ['/Users/zcr545/Desktop/Projects/repos/yucca_data/preprocessed/Task001_OASIS/demo/OASIS_1006', '/Users/zcr545/Desktop/Projects/repos/yucca_data/preprocessed/Task001_OASIS/demo/OASIS_1007', '/Users/zcr545/Desktop/Projects/repos/yucca_data/preprocessed/Task001_OASIS/demo/OASIS_1017']\n",
      "\n",
      "  | Name          | Type             | Params\n",
      "---------------------------------------------------\n",
      "0 | train_metrics | MetricCollection | 0     \n",
      "1 | val_metrics   | MetricCollection | 0     \n",
      "2 | model         | TinyUNet         | 7.6 K \n",
      "3 | loss_fn_train | DiceCE           | 0     \n",
      "4 | loss_fn_val   | DiceCE           | 0     \n",
      "---------------------------------------------------\n",
      "7.6 K     Trainable params\n",
      "0         Non-trainable params\n",
      "7.6 K     Total params\n",
      "0.030     Total estimated model params size (MB)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sanity Checking: |          | 0/? [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/zcr545/miniconda3/envs/yuccaenv/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:436: Consider setting `persistent_workers=True` in 'val_dataloader' to speed up the dataloader worker initialization.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sanity Checking DataLoader 0:   0%|          | 0/2 [00:00<?, ?it/s]torch.Size([2, 3, 32, 32]) torch.Size([2, 1, 32, 32])\n",
      "Sanity Checking DataLoader 0:  50%|█████     | 1/2 [00:00<00:00, 17.28it/s]torch.Size([2, 3, 32, 32]) torch.Size([2, 1, 32, 32])\n",
      "                                                                           "
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:root:Starting training with data from: /Users/zcr545/Desktop/Projects/repos/yucca_data/preprocessed/Task001_OASIS/demo\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/zcr545/miniconda3/envs/yuccaenv/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:436: Consider setting `persistent_workers=True` in 'train_dataloader' to speed up the dataloader worker initialization.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 0:   0%|          | 0/2 [00:00<?, ?it/s] torch.Size([2, 3, 32, 32]) torch.Size([2, 1, 32, 32])\n",
      "Epoch 0:  50%|█████     | 1/2 [00:03<00:03,  0.30it/s, v_num=0]torch.Size([2, 3, 32, 32]) torch.Size([2, 1, 32, 32])\n",
      "Epoch 0: 100%|██████████| 2/2 [00:03<00:00,  0.59it/s, v_num=0]torch.Size([2, 3, 32, 32]) torch.Size([2, 1, 32, 32])\n",
      "torch.Size([2, 3, 32, 32]) torch.Size([2, 1, 32, 32])\n",
      "Epoch 1:   0%|          | 0/2 [00:00<?, ?it/s, v_num=0]        torch.Size([2, 3, 32, 32]) torch.Size([2, 1, 32, 32])\n",
      "Epoch 1:  50%|█████     | 1/2 [00:35<00:35,  0.03it/s, v_num=0]torch.Size([2, 3, 32, 32]) torch.Size([2, 1, 32, 32])\n",
      "Epoch 1: 100%|██████████| 2/2 [00:35<00:00,  0.06it/s, v_num=0]torch.Size([2, 3, 32, 32]) torch.Size([2, 1, 32, 32])\n",
      "torch.Size([2, 3, 32, 32]) torch.Size([2, 1, 32, 32])\n",
      "Epoch 1: 100%|██████████| 2/2 [01:20<00:00,  0.02it/s, v_num=0]"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "`Trainer.fit` stopped: `max_epochs=2` reached.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1: 100%|██████████| 2/2 [01:20<00:00,  0.02it/s, v_num=0]\n"
     ]
    }
   ],
   "source": [
    "trainer = L.Trainer(\n",
    "    callbacks=callback_config.callbacks,\n",
    "    default_root_dir=path_config.save_dir,\n",
    "    limit_train_batches=2,\n",
    "    limit_val_batches=2,\n",
    "    log_every_n_steps=2,\n",
    "    logger=callback_config.loggers,\n",
    "    precision=\"32\",\n",
    "    profiler=callback_config.profiler,\n",
    "    enable_progress_bar=True,\n",
    "    max_epochs=2,\n",
    "    accelerator=\"cpu\",\n",
    ")\n",
    "\n",
    "\n",
    "trainer.fit(\n",
    "    model=model_module,\n",
    "    datamodule=data_module,\n",
    "    ckpt_path=\"last\",\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "yuccaenv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
