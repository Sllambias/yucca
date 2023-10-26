"""
CURRENTLY NOT MODIFIED FOR THE MIGRATION TO YUCCA
"""


import numpy as np
import torch
import torch.nn as nn
import wandb
import yucca
from torch import autocast
from torch.cuda.amp import GradScaler
from sklearn.model_selection import KFold
from yucca.training.trainers.YuccaTrainerV2 import YuccaTrainerV2
from yucca.paths import yucca_preprocessed
from yucca.image_processing.matrix_ops import get_max_rotated_size
from yucca.training.data_loading.YuccaLoader import YuccaLoader
from yucca.training.data_loading.alternative_loaders.YuccaLoader_Classification import YuccaLoader_Classification
from yucca.training.data_loading.alternative_loaders.YuccaLoader_NoSeg import YuccaLoader_NoSeg
from yucca.training.augmentation.YuccaAugmenterV2 import YuccaAugmenterV2
from yucca.training.augmentation.default_augmentation_params import default_3D_augmentation_paramsV2,\
    default_2D_augmentation_paramsV2
from yucca.training.augmentation.alternative_params.classification_augmentation_params import classification_2D_augmentation_params,\
    classification_3D_augmentation_params
from yucca.training.augmentation.alternative_params.reconstruction_augmentation_params import reconstruction_3D_augmentation_params,\
    reconstruction_2D_augmentation_params
from yucca.utils.files_and_folders import recursive_find_python_class, save_segmentation_from_logits
from yucca.utils.torch_utils import maybe_to_cuda
from yucca.training.loss_functions.CE import CE
from yucca.training.loss_functions.MSE import MSE
from yucca.utils.kwargs import filter_kwargs
from batchgenerators.utilities.file_and_folder_operations import save_json, join, load_json, \
    load_pickle, maybe_mkdir_p, isfile, subfiles, save_pickle



class YuccaTrainer_ResNet50(YuccaTrainer):
    """
    The difference from YuccaTrainerV2 --> YuccaTrainerV3 is:
    - Introduces Deep Supervision
    - Uses data augmentation scheme V2
    """
    def __init__(self, model,
                 model_dimensions: str,
                 task: str,
                 folds: str | int,
                 plan_id: str,
                 starting_lr: float = None,
                 loss_fn: str = None,
                 momentum: float = None,
                 continue_training: bool = False,
                 checkpoint: str = None,
                 finetune: bool = False,
                 fast_training: bool = False):
        super().__init__(model, model_dimensions, task, folds, plan_id, starting_lr, loss_fn,
                         momentum, continue_training, checkpoint, finetune, fast_training)
        self._DEFAULT_LOSS = {"Classification": CE,
                            "Reconstruction": MSE,
                            "Segmentation": CE}
    
        
    def get_data_generators(self):
        # Here we wrap the augmenters so we can call one instance which will randomly select one 
        # each time next(self.tr_gen) is called
        self.cls_tr_gen, self.cls_val_gen = YuccaAugmenterV2(self.cls_tr_loader, self.cls_val_loader,
                                                          self.patch_size,
                                                          self.cls_augmentation_parameters)
                                                  
        self.re_tr_gen, self.re_val_gen = YuccaAugmenterV2(self.re_tr_loader, self.re_val_loader,
                                                        self.patch_size,
                                                        self.re_augmentation_parameters)

        self.seg_tr_gen, self.seg_val_gen = YuccaAugmenterV2(self.seg_tr_loader, self.seg_val_loader,
                                                          self.patch_size,
                                                          self.seg_augmentation_parameters)

    def initialize_network(self):

        self.network = recursive_find_python_class(folder=[join(yucca.__path__[0], 'architectures')],
                                                   class_name=self.model_name,
                                                   current_module='yucca.architectures')

        if self.model_dimensions == '3D':
            conv = nn.Conv3d
            norm = nn.InstanceNorm3d
            dropout = nn.Dropout3d
        else:
            conv = nn.Conv2d
            norm = nn.InstanceNorm2d
            dropout = nn.Dropout2d

        self.network = self.network(input_channels=self.nmodalities,
                                    num_cls_classes=len(self.classes[0]),
                                    num_seg_classes=len(self.classes[2]),
                                    conv_op=conv,
                                    norm_op=norm,
                                    dropout_op=dropout,
                                    patch_size=self.patch_size,
                                    deep_supervision=self.deep_supervision)

        if torch.cuda.is_available():
            self.network.cuda()

    def initialize_loss_optim_lr(self):
        self.grad_scaler = GradScaler(enabled=True)
        # Defining the loss
        self.loss_fn = self._DEFAULT_LOSS
        assert isinstance(self.loss_fn, dict), "expected losses to be a dict of losses"
        assert len(self.tasks) == len(self.loss_fn), "expected n losses to be equal to n tasks"

        self.loss_fn_kwargs = {
                            'soft_dice_kwargs': {'apply_softmax': True},    # DCE
                            'hierarchical_kwargs': {'rootdict': self.plans['dataset_properties']['label_hierarchy']}, # Hierarchical Loss
                             }

        for task, loss_fn in self.loss_fn.items():
            kwargs = filter_kwargs(loss_fn, self.loss_fn_kwargs)
            self.loss_fn[task] = loss_fn(**kwargs)
            self.log(f'{f"{task} loss function":20}', self.loss_fn[task].__class__.__name__, time=False)

        # Defining the learning rate
        if not self.starting_lr:
            self.starting_lr = self._DEFAULT_STARTING_LR
        if self.finetune:
            self.starting_lr /= 10
        self.log(f'{"learning rate:":20}', self.starting_lr, time=False)

        # Defining the momentum
        if not self.momentum:
            self.momentum = self._DEFAULT_MOMENTUM
        self.log(f'{"momentum":20}', self.momentum, time=False)

        # And constructing the optimizer
        # Set kwargs for all optimizers and then filter relevant ones based on optimizer class

        self.optim_kwargs = {
                             'lr': float(self.starting_lr),                 # all
                             'momentum': float(self.momentum),              # SGD
                             'eps': 1e-4,
                             'weight_decay': 3e-5,
                             }
        self.optim_kwargs = filter_kwargs(self.optim, self.optim_kwargs)
        self.optim = self.optim(self.network.parameters(), **self.optim_kwargs)
        self.log(f'{"optimizer":20}', self.optim.__class__.__name__, time=False)

        # Set kwargs for all schedulers and then filter relevant ones based on scheduler class
        self.lr_scheduler_kwargs = {
            'T_max': self.final_epoch, 'eta_min': 1e-9,                     # Cosine Annealing
            }
        self.lr_scheduler_kwargs = filter_kwargs(self.lr_scheduler, self.lr_scheduler_kwargs)
        self.lr_scheduler = self.lr_scheduler(self.optim, **self.lr_scheduler_kwargs)
        self.log(f'{"LR scheduler":20}', self.lr_scheduler.__class__.__name__, time=False)

    def load_data(self):
        self.tasks = list(self.plans['dataset_properties']['tasks'].keys())

        assert len(self.patch_size) in [2, 3], "Patch Size should be (x, y, z) or (x, y)"\
            f" but is: {self.patch_size}"
       
        if not self.is_seeded:
            self.set_random_seeds()

        self.folder_with_preprocessed_data = join(yucca_preprocessed, self.task,
                                                  self.plan_id)
        self.splits_file = join(yucca_preprocessed, self.task, 'splits.pkl')
        self.log(f'{"data folder:":20}', self.folder_with_preprocessed_data, time=False)

        if not isfile(self.splits_file):
            self.split_data()

        self.splits = load_pickle(self.splits_file)[self.folds]

        self.cls_train_samples = [join(self.folder_with_preprocessed_data, sample) for sample in self.splits['train'] if sample.split(".")[0] in self.plans['dataset_properties']['tasks']['Classification']]
        self.cls_val_samples = [join(self.folder_with_preprocessed_data, sample) for sample in self.splits['val'] if sample.split(".")[0] in self.plans['dataset_properties']['tasks']['Classification']]
        self.re_train_samples = [join(self.folder_with_preprocessed_data, sample) for sample in self.splits['train'] if sample.split(".")[0] in self.plans['dataset_properties']['tasks']['Reconstruction']]
        self.re_val_samples = [join(self.folder_with_preprocessed_data, sample) for sample in self.splits['val'] if sample.split(".")[0] in self.plans['dataset_properties']['tasks']['Reconstruction']]
        self.seg_train_samples = [join(self.folder_with_preprocessed_data, sample) for sample in self.splits['train'] if sample.split(".")[0] in self.plans['dataset_properties']['tasks']['Segmentation']]
        self.seg_val_samples = [join(self.folder_with_preprocessed_data, sample) for sample in self.splits['val'] if sample.split(".")[0] in self.plans['dataset_properties']['tasks']['Segmentation']]

        self.train_samples = self.cls_train_samples + self.re_train_samples + self.seg_train_samples
        self.val_samples = self.cls_val_samples + self.re_val_samples + self.seg_val_samples

        """Here we want to calculate a larger-than-final patch size
        to avoid cropping out parts that would be rotated into inclusion
        and to avoid large interpolation artefacts near the borders of our final patch
        For this we use the hypotenuse of the 2 largest dimensions"""
        self.initial_patch_size = get_max_rotated_size(self.patch_size)

        self.log(f'{"patch size pre-aug:":20}', self.initial_patch_size, time=False)
        self.log(f'{"patch size final:":20}', self.patch_size, time=False)
        self.log(f'{"batch size:":20}', self.batch_size, time=False)
        self.log(f'{"classes":20}', self.classes, time=False)
        self.log(f'{"modalities:":20}', self.modalities, time=False)
        self.log(f'{"fold:":20}', self.folds, time=False)
        self.log(f'{"train batches/epoch:":20}', self.train_batches_per_epoch, time=False)
        self.log(f'{"val batches/epoch:":20}', self.val_batches_per_epoch, time=False)

        self.cls_tr_loader = YuccaLoader_Classification(self.cls_train_samples,
                                                      self.batch_size,
                                                      self.initial_patch_size,
                                                      p_oversample_foreground=self.p_force_foreground
                                                      )
        self.cls_val_loader = YuccaLoader_Classification(self.cls_val_samples,
                                                       self.batch_size,
                                                       self.patch_size,
                                                       p_oversample_foreground=self.p_force_foreground
                                                       )
        
        self.re_tr_loader = YuccaLoader_NoSeg(self.re_train_samples,
                                            self.batch_size,
                                            self.initial_patch_size,
                                            p_oversample_foreground=self.p_force_foreground)
        self.re_val_loader = YuccaLoader_NoSeg(self.re_val_samples,
                                     self.batch_size,
                                     self.patch_size,
                                     p_oversample_foreground=self.p_force_foreground)
        
        self.seg_tr_loader = YuccaLoader(self.seg_train_samples,
                                    self.batch_size,
                                    self.initial_patch_size,
                                    p_oversample_foreground=self.p_force_foreground)
        self.seg_val_loader = YuccaLoader(self.seg_val_samples,
                                     self.batch_size,
                                     self.patch_size,
                                     p_oversample_foreground=self.p_force_foreground)

        self.get_data_generators()


    def run_batch(self, batch, task: str, train=True, comprehensive_eval=False, debug=False):
        self.optim.zero_grad()
        image = batch['image']
        seg = batch['seg']

        image = maybe_to_cuda(image)
        seg = maybe_to_cuda(seg)

        with autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            pred = self.network(image, task)
            del image
            loss = self.loss_fn[task](pred, seg)
        if comprehensive_eval:
            # Add additional_eval here to retrieve e.g. Dice score, #TP, #FP, etc.
            self.comprehensive_eval(pred, seg)
        del seg

        if train:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optim)
            # The max norm here is a mythical setting with limited documentation/experiments. Just trust.
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=12)
            self.grad_scaler.step(self.optim)
            self.grad_scaler.update()

        return loss.detach().cpu().numpy()
    
    def run_training(self):
        self.initialize()

        _ = self.cls_tr_gen.next()
        _ = self.cls_val_gen.next()
        _ = self.re_tr_gen.next()
        _ = self.re_val_gen.next()
        _ = self.seg_tr_gen.next()
        _ = self.seg_val_gen.next()

        self.tr_gen = {'Classification': self.cls_tr_gen,
                       'Reconstruction': self.re_tr_gen,
                       'Segmentation': self.seg_tr_gen}
        self.val_gen = {'Classification': self.cls_val_gen,
                        'Reconstruction': self.re_val_gen,
                        'Segmentation': self.seg_val_gen}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            self.log("CUDA NOT AVAILABLE. YOU SHOULD REALLY NOT TRAIN WITHOUT CUDA!", time=False)

        while self.current_epoch < self.final_epoch:
            self.epoch_start()

            self.network.train()
            for _ in range(self.train_batches_per_epoch):
                task = np.random.choice(self.tasks)
                batch_loss = self.run_batch(next(self.tr_gen[task]), task)
                self.epoch_tr_loss.append(batch_loss)

            self.network.eval()
            with torch.no_grad():
                for _ in range(self.val_batches_per_epoch):
                    task = np.random.choice(self.tasks)
                    batch_loss = self.run_batch(next(self.val_gen[task]), task, train=False, comprehensive_eval=False, debug=False)
                    self.epoch_val_loss.append(batch_loss)

            self.tr_losses.append(np.mean(self.epoch_tr_loss))
            self.val_losses.append(np.mean(self.epoch_val_loss))

            if self.lr_scheduler:
                self.lr_scheduler.step()

            self.epoch_finish()

        if self.current_epoch == self.final_epoch:
            self.save_checkpoint("checkpoint_final.model")

    def set_batch_and_patch_sizes(self):
        self.batch_size = 128
        self.patch_size = (224, 224)

    def setup_DA(self):
        if self.model_dimensions == '3D':
            self.cls_augmentation_parameters = classification_3D_augmentation_params
            self.re_augmentation_parameters = reconstruction_3D_augmentation_params
            self.seg_augmentation_parameters = default_3D_augmentation_paramsV2
        if self.model_dimensions in ['2D', '25D']:
            self.cls_augmentation_parameters = classification_2D_augmentation_params
            self.re_augmentation_parameters = reconstruction_2D_augmentation_params
            self.seg_augmentation_parameters = default_2D_augmentation_paramsV2

    def split_data(self):
        splits = []
        suffix = '.npy'
        files = subfiles(self.folder_with_preprocessed_data, join=False, suffix='.npy')
        if not files:
            files = subfiles(self.folder_with_preprocessed_data, join=False, suffix='.npz')
            if files:
                suffix = '.npz'
                self.log("Only found compressed (.npz) files. This might increase runtime.",
                         time=False)

        assert files, f"Couldn't find any .npy or .npz files in {self.folder_with_preprocessed_data}"

        v = np.array([i+suffix for i in self.plans['dataset_properties']['tasks'][self.tasks[0]]])
        kf = KFold(n_splits=5, shuffle=True)
        for train, val in kf.split(v):
            splits.append({'train': list(v[train]), 'val':list(v[val])})
        for task in self.tasks[1:]:
            v = np.array([i+suffix for i in self.plans['dataset_properties']['tasks'][task]])
            kf = KFold(n_splits=5, shuffle=True)
            for idx, train_val in enumerate(kf.split(v)):
                train, val = train_val
                splits[idx]['train'].extend(list(v[train]))
                splits[idx]['val'].extend(list(v[val]))
        save_pickle(splits, self.splits_file)

 