import numpy as np
import os
import torch
import wandb
import sys
import random
import yucca
from time import localtime, strftime, time, mktime
from abc import abstractmethod
from collections import OrderedDict
from torch import nn, autocast
from batchgenerators.utilities.file_and_folder_operations import (
    save_json,
    join,
    load_json,
    load_pickle,
    maybe_mkdir_p,
    isfile,
    subfiles,
)
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.transforms.abstract_transforms import Compose
from yucca.paths import yucca_preprocessed_data, yucca_models
from yucca.training.data_loading.YuccaLoader import YuccaLoader
from yucca.preprocessing.YuccaPreprocessor import YuccaPreprocessor
from yucca.image_processing.matrix_ops import get_max_rotated_size
from yucca.network_architectures.utils.model_memory_estimation import (
    find_optimal_tensor_dims,
)
from yucca.utils.files_and_folders import recursive_find_python_class
from yucca.utils.saving import save_prediction_from_logits
from yucca.utils.torch_utils import maybe_to_gpu
from yucca.image_processing.transforms.BiasField import BiasField
from yucca.image_processing.transforms.Blur import Blur
from yucca.image_processing.transforms.CopyImageToSeg import CopyImageToSeg
from yucca.image_processing.transforms.cropping_and_padding import CropPad
from yucca.image_processing.transforms.formatting import NumpyToTorch
from yucca.image_processing.transforms.Gamma import Gamma
from yucca.image_processing.transforms.Ghosting import MotionGhosting
from yucca.image_processing.transforms.Masking import Masking
from yucca.image_processing.transforms.Mirror import Mirror
from yucca.image_processing.transforms.Noise import (
    AdditiveNoise,
    MultiplicativeNoise,
)
from yucca.image_processing.transforms.Ringing import GibbsRinging
from yucca.image_processing.transforms.sampling import DownsampleSegForDS
from yucca.image_processing.transforms.SimulateLowres import SimulateLowres
from yucca.image_processing.transforms.Spatial import Spatial


class base_manager(object):
    def __init__(self):
        self.train_command = None
        self.is_initialized = False
        self.log_file = None
        self.param_dict = OrderedDict()
        self.network = None
        self.final_epoch = 1000
        self.current_epoch = 0
        self.is_seeded = False
        self.save_every = 300
        self.p_force_foreground = 0.33
        self.deep_supervision = False
        self.max_vram = 12

        # Debugging
        self.get_timings = True

        # These can be changed during inference
        self.threads_for_inference = 2
        self.threads_for_augmentation = 6
        self.sliding_window_overlap = 0.5
        self.preprocessor = None
        self.compression_level = 9  # 0-9 with 0 = min and 9 = max

        # These will always be set by the individual Trainer
        # Using arguments supplied by run_training
        self.train_batches_per_epoch = self.val_batches_per_epoch = self.batch_size_2D = self.batch_size_3D = self.folds = (
            self.model_dimensions
        ) = self.model_name = self.outpath = self.patch_size_2D = self.patch_size_3D = self.task = self.plan_id = self.name = (
            None
        )

        # These can be set by the individual Trainer
        # Using optional arguments supplied by run_training
        self.starting_lr = self.grad_scaler = self.loss_fn = self.loss_fn_kwargs = self.lr_scheduler = (
            self.lr_scheduler_kwargs
        ) = self.momentum = self.optim = self.optim_kwargs = self.random_seed = self.fast_training = self.finetune = (
            self.fast_train_batches_per_epoch
        ) = self.fast_val_batches_per_epoch = self.fast_final_epoch = None

        # These will always be set by the plans file
        self.classes = self.nclasses = self.modalities = self.nmodalities = self.folder_with_preprocessed_data = self.plans = (
            self.plans_path
        ) = None

        # These will be used during training
        self.tr_losses = []
        self.val_losses = []
        self.best_val_loss = 99999

    @abstractmethod
    def comprehensive_eval(self):
        """
        implement in trainer subclass
        """
        pass

    @abstractmethod
    def initialize(self):
        """
        implement in trainer subclass
        """
        pass

    @abstractmethod
    def initialize_loss_optim_lr(self):
        """
        implement in trainer subclass
        """
        pass

    @abstractmethod
    def run_training(self):
        """
        implement this in subclasses
        """
        pass

    def setup_DA(self):
        # Define whether we crop before or after applying augmentations
        # Define if cropping is random or always centered
        self.EarlyCrop = True
        self.RandomCrop = True
        self.MaskImageForReconstruction = False

        # Label/segmentation transforms
        self.SkipSeg = False
        self.SegDtype = int
        self.CopyImageToSeg = False

        self.AdditiveNoise_p_per_sample = 0.0
        self.AdditiveNoise_mean = (0.0, 0.0)
        self.AdditiveNoise_sigma = (1e-3, 1e-4)

        self.BiasField_p_per_sample = 0.0

        self.Blurring_p_per_sample = 0.0
        self.Blurring_sigma = (0.0, 1.0)
        self.Blurring_p_per_channel = 0.5

        self.ElasticDeform_p_per_sample = 0.0
        self.ElasticDeform_alpha = (200, 600)
        self.ElasticDeform_sigma = (20, 30)

        self.Gamma_p_per_sample = 0.0
        self.Gamma_p_invert_image = 0.05
        self.Gamma_range = (0.5, 2.0)

        self.GibbsRinging_p_per_sample = 0.0
        self.GibbsRinging_cutFreq = (96, 129)
        self.GibbsRinging_axes = (0, 3)

        self.Mirror_p_per_sample = 0.0
        self.Mirror_p_per_axis = 0.33
        self.Mirror_axes = (0, 1, 2)

        self.MotionGhosting_p_per_sample = 0.0
        self.MotionGhosting_alpha = (0.85, 0.95)
        self.MotionGhosting_numReps = (2, 11)
        self.MotionGhosting_axes = (0, 3)

        self.MultiplicativeNoise_p_per_sample = 0.0
        self.MultiplicativeNoise_mean = (0, 0)
        self.MultiplicativeNoise_sigma = (1e-3, 1e-4)

        self.Rotation_p_per_sample = 0.0
        self.Rotation_p_per_axis = 0.66
        self.Rotation_x = (-30.0, 30.0)
        self.Rotation_y = (-30.0, 30.0)
        self.Rotation_z = (-30.0, 30.0)

        self.Scale_p_per_sample = 0.0
        self.Scale_factor = (0.9, 1.1)

        self.SimulateLowres_p_per_sample = 0.0
        self.SimulateLowres_p_per_channel = 0.5
        self.SimulateLowres_p_per_axis = 0.33
        self.SimulateLowres_zoom_range = (0.5, 1.0)

        if self.model_dimensions == "2D":
            self.GibbsRinging_axes = (0, 2)
            self.Mirror_axes = (0, 1)
            self.MotionGhosting_axes = (0, 2)
            self.Rotation_y = (-0.0, 0.0)
            self.Rotation_z = (-0.0, 0.0)

    @abstractmethod
    def split_data(self):
        """
        implement in trainer subclass
        """
        pass

    def epoch_finish(self):
        self.epoch_end_time = time()
        self.log(f'{"Current Epoch:":20}', self.current_epoch)
        self.log(f'{"Training Loss:":20}', round(self.tr_losses[-1], 3))
        self.log(f'{"Validation Loss:":20}', round(self.val_losses[-1], 3))
        wandb.log(
            {
                "Training Loss": round(self.tr_losses[-1], 3),
                "Validation Loss": round(self.val_losses[-1], 3),
            },
            commit=False,
        )
        # self.log(f'{"Validation Dice:":15}', )
        # self.log(f'{":":15}', )

        if self.epoch_eval_dict:
            epoch_eval_dict_per_label = {}
            for key in self.epoch_eval_dict:
                self.epoch_eval_dict[key] = np.round(np.nanmean(self.epoch_eval_dict[key], 0), 4)
                self.log(f"{key:20}", self.epoch_eval_dict[key])
                for i, val in enumerate(self.epoch_eval_dict[key]):
                    epoch_eval_dict_per_label[f"{key}_{i+1}"] = val
            wandb.log(epoch_eval_dict_per_label, commit=False)

        if round(self.val_losses[-1], 3) < self.best_val_loss:
            self.best_val_loss = round(self.val_losses[-1], 3)
            wandb.log({"Best Validation Loss": self.best_val_loss}, commit=False)
            self.save_checkpoint("checkpoint_best.model")

        if self.current_epoch % self.save_every == 0:
            self.save_checkpoint(f"checkpoint_{self.current_epoch}.model")
        elif self.current_epoch % 50 == 0:
            self.save_checkpoint(f"checkpoint_latest.model")

        wandb.log({"Learning Rate": self.optim.param_groups[0]["lr"]}, commit=False)
        self.log(f'{"Learning Rate:":20}', self.optim.param_groups[0]["lr"])

        self.epoch_time = self.epoch_end_time - self.epoch_start_time
        self.log(f'{"Time elapsed:":20}', self.epoch_time)
        # This is THE log call that will update the "step" counter in wandb.
        # All others wandb.log calls (should) have commit=False.
        wandb.log({"Time/epoch": self.epoch_end_time - self.epoch_start_time})

        self.current_epoch += 1
        sys.stdout.flush()

    def epoch_start(self):
        self.epoch_tr_loss = []
        self.epoch_val_loss = []
        self.epoch_eval_dict = None
        self.epoch_start_time = time()
        self.log("\n", time=False, also_print=True)

    def initialize_network(self):
        self.network = recursive_find_python_class(
            folder=[join(yucca.__path__[0], "network_architectures")],
            class_name=self.model_name,
            current_module="yucca.network_architectures",
        )

        if self.model_dimensions == "3D":
            conv = nn.Conv3d
            norm = nn.InstanceNorm3d
            dropout = nn.Dropout3d
        else:
            conv = nn.Conv2d
            norm = nn.InstanceNorm2d
            dropout = nn.Dropout2d

        self.network = self.network(
            input_channels=self.nmodalities,
            num_classes=self.nclasses,
            conv_op=conv,
            norm_op=norm,
            dropout_op=dropout,
            patch_size=self.patch_size,
            deep_supervision=self.deep_supervision,
        )

        if torch.cuda.is_available():
            self.network.cuda()

    def initialize_preprocessor_for_inference(self):
        preprocessor_class = recursive_find_python_class(
            folder=[join(yucca.__path__[0], "preprocessing")],
            class_name=self.plans["preprocessor"],
            current_module="yucca.preprocessing",
        )
        assert preprocessor_class, f"searching for {self.plans['preprocessor']}" f"but found: {preprocessor_class}"
        assert issubclass(preprocessor_class, YuccaPreprocessor), "Preprocessor is not a subclass " "of YuccaPreprocesor"
        print(f"{'Using preprocessor: ':25} {preprocessor_class}")
        self.preprocessor = preprocessor_class(self.plans_path)

    def initialize_wandb(self):
        wandb.init(
            project="Yucca",
            group=self.task,
            dir=self.outpath,
            name=join(os.path.split(self.outpath)[-1] + "_" + strftime("%Y_%m_%d_%H_%M_%S", self.initial_time_obj)),
            config=self.param_dict,
        )

    def load_checkpoint(self, checkpoint, train=False):
        """
        If we are loading a checkpoint to continue training or finetune etc., then we need to
        initialize() to have outpaths, training data, logging and data augmentation

        If we are loading a checkpoint for inference we just need to initialize the network
        and then apply the pretrained weights
        """

        self.load_plans_from_path(checkpoint + ".json")

        chk = torch.load(checkpoint, map_location=torch.device("cpu"))

        if self.finetune:
            # Filter out layers from old model if they dont match sizes of the new model
            # e.g. when using a model pretrained on 1 label for a multi-label problem.
            # this will discard the weights final layer
            state_dict = {
                k: v
                for k, v in chk["model_state_dict"].items()
                if k in self.network.state_dict().keys() and v.shape == self.network.state_dict()[k].shape
            }
            # Then make sure optimizer layers are appropriate, otherwise re-initialize them
            # with appropriate dimensions
            for idx, layer in enumerate(self.network.parameters()):
                if layer.shape != chk["optimizer_state_dict"]["state"][idx]["momentum_buffer"]:
                    chk["optimizer_state_dict"]["state"][idx]["momentum_buffer"] = torch.zeros(layer.shape)
        else:
            state_dict = chk["model_state_dict"]

        if "patch_size" in chk:
            self.patch_size = chk["patch_size"]
        else:
            print(
                "patch_size not found in checkpoint. Generating a new one now, better hope "
                "it's the same as the training patch_size."
            )
            self.set_batch_and_patch_sizes()

        if train:
            self.optim.load_state_dict(chk["optimizer_state_dict"])
            self.current_epoch = chk["epoch"]
            self.tr_losses = chk["tr_losses"]
            self.val_losses = chk["val_losses"]
            self.best_val_loss = chk["best"]
            if self.grad_scaler is not None:
                self.grad_scaler.load_state_dict(chk["grad_scaler_state_dict"])
            self.log("Continuing training from checkpoint:", checkpoint, time=False)
            if self.finetune:
                self.best_val_loss = 99999
                self.final_epoch = 500
                self.current_epoch = 0
                for param in self.optim.param_groups:
                    param["lr"] = self._DEFAULT_STARTING_LR * 0.1
        else:
            self.initialize_network()

        self.network.load_state_dict(state_dict, strict=False)

        if torch.cuda.is_available():
            self.network.cuda()

    def load_data(self):
        assert len(self.patch_size) in [2, 3], "Patch Size should be (x, y, z) or (x, y)" f" but is: {self.patch_size}"

        if not self.is_seeded:
            self.set_random_seeds()

        self.folder_with_preprocessed_data = join(yucca_preprocessed_data, self.task, self.plan_id)
        self.splits_file = join(yucca_preprocessed_data, self.task, "splits.pkl")
        self.log(f'{"data folder:":20}', self.folder_with_preprocessed_data, time=False)

        if not isfile(self.splits_file):
            self.split_data()

        self.splits = load_pickle(self.splits_file)[self.folds]

        self.train_samples = [join(self.folder_with_preprocessed_data, sample) for sample in self.splits["train"]]
        self.val_samples = [join(self.folder_with_preprocessed_data, sample) for sample in self.splits["val"]]
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

        self.tr_loader = YuccaLoader(
            self.train_samples,
            self.batch_size,
            self.initial_patch_size,
            p_oversample_foreground=self.p_force_foreground,
        )
        self.val_loader = YuccaLoader(
            self.val_samples,
            self.batch_size,
            self.patch_size,
            p_oversample_foreground=self.p_force_foreground,
        )

        self.get_data_generators()

    def get_data_generators(self):
        train_transforms = []
        val_transforms = []

        # Augmentations
        train_transforms.append(
            Spatial(
                crop=self.EarlyCrop,
                random_crop=self.RandomCrop,
                patch_size=self.patch_size,
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
                skip_seg=self.SkipSeg,
            )
        )

        train_transforms.append(
            AdditiveNoise(
                p_per_sample=self.AdditiveNoise_p_per_sample,
                mean=self.AdditiveNoise_mean,
                sigma=self.AdditiveNoise_sigma,
            )
        )

        train_transforms.append(
            Blur(
                p_per_sample=self.Blurring_p_per_sample,
                p_per_channel=self.Blurring_p_per_channel,
                sigma=self.Blurring_sigma,
            )
        )

        train_transforms.append(
            MultiplicativeNoise(
                p_per_sample=self.MultiplicativeNoise_p_per_sample,
                mean=self.MultiplicativeNoise_mean,
                sigma=self.MultiplicativeNoise_sigma,
            )
        )

        train_transforms.append(
            MotionGhosting(
                p_per_sample=self.MotionGhosting_p_per_sample,
                alpha=self.MotionGhosting_alpha,
                numReps=self.MotionGhosting_numReps,
                axes=self.MotionGhosting_axes,
            )
        )

        train_transforms.append(
            GibbsRinging(
                p_per_sample=self.GibbsRinging_p_per_sample,
                cutFreq=self.GibbsRinging_cutFreq,
                axes=self.GibbsRinging_axes,
            )
        )

        train_transforms.append(
            SimulateLowres(
                p_per_sample=self.SimulateLowres_p_per_sample,
                p_per_channel=self.SimulateLowres_p_per_channel,
                p_per_axis=self.SimulateLowres_p_per_axis,
                zoom_range=self.SimulateLowres_zoom_range,
            )
        )

        train_transforms.append(BiasField(p_per_sample=self.BiasField_p_per_sample))

        train_transforms.append(
            Gamma(
                p_per_sample=self.Gamma_p_per_sample,
                p_invert_image=self.Gamma_p_invert_image,
                gamma_range=self.Gamma_range,
            )
        )

        train_transforms.append(
            Mirror(
                p_per_sample=self.Mirror_p_per_sample,
                axes=self.Mirror_axes,
                p_mirror_per_axis=self.Mirror_p_per_axis,
                skip_seg=self.SkipSeg,
            )
        )

        if not self.EarlyCrop:
            train_transforms.append(CropPad(patch_size=self.patch_size, random_crop=self.RandomCrop))

        if self.deep_supervision:
            train_transforms.append(DownsampleSegForDS())

        if self.CopyImageToSeg:
            train_transforms.append(CopyImageToSeg())

        if self.MaskImageForReconstruction:
            train_transforms.append(Masking())

        train_transforms.append(NumpyToTorch(seg_dtype=self.SegDtype))
        train_transforms = Compose(train_transforms)

        self.tr_gen = NonDetMultiThreadedAugmenter(
            self.tr_loader,
            train_transforms,
            self.threads_for_augmentation,
            2,
            pin_memory=True,
        )

        # Validation Transforms
        if self.deep_supervision:
            val_transforms.append(DownsampleSegForDS())

        if self.CopyImageToSeg:
            val_transforms.append(CopyImageToSeg())

        if self.MaskImageForReconstruction:
            val_transforms.append(Masking())

        val_transforms.append(NumpyToTorch(seg_dtype=self.SegDtype))
        val_transforms = Compose(val_transforms)
        self.val_gen = NonDetMultiThreadedAugmenter(
            self.val_loader,
            val_transforms,
            self.threads_for_augmentation // 2,
            2,
            pin_memory=True,
        )

    def load_plans_from_path(self, path):
        # if plans already exist, it is because we are finetuning.
        # In this case we want to combine plans from two tasks, keeping dataset properties
        # from the already loaded plans, and the rest (such as normalization, tranposition etc.,)
        # from the one we are loading afterwards.
        self.plans_path = path

        if self.plans:
            properties = self.plans["dataset_properties"]
            self.plans = load_json(self.plans_path)
            self.plans["dataset_properties"] = properties
        else:
            self.plans = load_json(self.plans_path)

        self.classes = self.plans["dataset_properties"]["classes"]
        self.nclasses = len(self.classes)

        self.modalities = self.plans["dataset_properties"]["modalities"]
        self.nmodalities = len(self.modalities)

        self.plans["trainer_class"] = self.__class__.__name__

        if self.model_dimensions is None:
            self.model_dimensions = self.plans["suggested_dimensionality"]

    def log(self, *args, time=True, also_print=True):
        if not self.log_file:
            self.initial_time_obj = localtime()
            self.log_file = join(
                self.outpath,
                "log_" + strftime("%Y_%m_%d_%H_%M_%S", self.initial_time_obj) + ".txt",
            )
            with open(self.log_file, "w") as f:
                f.write("Starting model training")
                print("Starting model training \n" f'{"log file:":20} {self.log_file} \n')
                f.write("\n")
                f.write(f'{"log file:":20} {self.log_file}')
                f.write("\n")
        t = strftime("%Y_%m_%d_%H_%M_%S", localtime())
        with open(self.log_file, "a+") as f:
            if time:
                f.write(t)
                f.write(" ")
            for arg in args:
                f.write(str(arg))
                f.write(" ")
            f.write("\n")
        if also_print:
            print(" ".join([str(arg) for arg in args]))

    def predict_folder(
        self,
        input_folder,
        output_folder,
        not_strict=True,
        save_softmax=False,
        overwrite=False,
        do_tta=False,
    ):
        self.initialize_preprocessor_for_inference()

        files = subfiles(input_folder, suffix=".nii.gz", join=False)

        if not not_strict:
            # If strict we enforce modality encoding.
            # This means files must be encoded as the model expects.
            # e.g. "_000" or "_001" etc., for T1 and T2 scans.
            # This allows us to handle different modalities
            # of the same subject, rather than treating them as individual cases.
            expected_modalities = self.plans["dataset_properties"]["modalities"]
            subject_ids, counts = np.unique([i[: -len("_000.nii.gz")] for i in files], return_counts=True)
            assert all(counts == len(expected_modalities)), "Aborting. Modalities are missing for some samples"
        else:
            subject_ids = np.unique([i[: -len(".nii.gz")] for i in files])

        all_cases = []
        all_outpaths = []
        for subject_id in subject_ids:
            case = [
                impath
                for impath in subfiles(input_folder, suffix=".nii.gz")
                if os.path.split(impath)[-1][: -len("_000.nii.gz")] == subject_id
            ]
            outpath = join(output_folder, subject_id)
            all_cases.append([case, outpath, save_softmax, overwrite, do_tta])
            all_outpaths.append(outpath)

        n_already_predicted = len(subfiles(output_folder, suffix=".nii.gz"))

        print(
            f"\n"
            f"STARTING PREDICTION \n"
            f"{'Cases already predicted: ':25} {n_already_predicted} \n"
            f"{'Cases NOT predicted: ':25} {len(all_outpaths) - n_already_predicted} \n"
            f"{'Overwrite predictions: ':25} {overwrite} \n"
        )

        for case_info in all_cases:
            self.preprocess_predict_test_case(*case_info)

    def preprocess_predict_test_case(self, case, outpath, save_softmax=False, overwrite=False, do_tta=False):
        """
        Load the preprocessor if not already loaded (e.g. in single prediction cases)
        Preprocessor will apply the same type of preprocessing that was applied to training data,
        and transpose to target view if necessary as well as handling data formats (torch GPU/CPU)
        Finally we detach, move tensor to GPU and convert to np.ndarray before transposing it
        to the original view.
        """

        # If overwrite = True, we don't care if file already exists
        # If overwrite = False, check if files already exist. If already exists we skip.

        if not overwrite:
            if isfile(outpath + ".nii.gz"):
                if not save_softmax:
                    return
                if save_softmax:
                    if isfile(outpath + ".npz"):
                        return

        if not self.preprocessor:
            self.initialize_preprocessor_for_inference()

        print(f"{'Predicting case: ':25} {case}")
        case, image_properties = self.preprocessor.preprocess_case_for_inference(case, self.patch_size)

        logits = (
            self.network.predict(
                mode=self.model_dimensions,
                data=case,
                patch_size=self.patch_size,
                overlap=self.sliding_window_overlap,
                mirror=do_tta,
            )
            .detach()
            .cpu()
            .numpy()
        )

        logits = self.preprocessor.reverse_preprocessing(logits, image_properties)

        print(f"{'Saving as: ':25} {outpath}")
        save_prediction_from_logits(
            logits,
            outpath=outpath,
            properties=image_properties,
            save_softmax=save_softmax,
            compression=self.compression_level,
        )
        print("\n")

    def run_batch(self, batch, train=True, comprehensive_eval=False):
        self.optim.zero_grad()

        image = batch["image"]
        seg = batch["seg"]

        image = maybe_to_gpu(image)
        seg = maybe_to_gpu(seg)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            pred = self.network(image)
            del image
            loss = self.loss_fn(pred, seg)

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

    def save_checkpoint(self, fname):
        self.log(f'{"Saving model:":20}', fname)
        model_dict = {
            "epoch": self.current_epoch,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "tr_losses": self.tr_losses,
            "val_losses": self.val_losses,
            "best": self.best_val_loss,
            "plans": self.plans_path,
            "patch_size": self.patch_size,
        }
        if self.grad_scaler is not None:
            model_dict["grad_scaler_state_dict"] = self.grad_scaler.state_dict()
        torch.save(model_dict, join(self.outpath, fname))
        save_json(self.plans, join(self.outpath, fname + ".json"))

    def save_parameter_json(self):
        # Model settings
        self.param_dict["network"] = self.model_name
        self.param_dict["model dimensions"] = self.model_dimensions

        # Data
        self.param_dict["patch_size"] = self.patch_size
        self.param_dict["plans id"] = self.plan_id
        self.param_dict["folds"] = self.folds
        self.param_dict["classes"] = self.classes
        self.param_dict["modalities"] = self.modalities
        self.param_dict["batch size"] = self.batch_size
        self.param_dict["max epochs"] = self.final_epoch
        self.param_dict["train batches per epoch"] = self.train_batches_per_epoch
        self.param_dict["val batches per epoch"] = self.val_batches_per_epoch
        self.param_dict["foreground oversampling"] = self.p_force_foreground
        self.param_dict["train_samples"] = self.train_samples
        self.param_dict["val_samples"] = self.val_samples

        # Parameters
        self.param_dict["loss function"] = self.loss_fn.__class__.__name__
        self.param_dict["loss kwargs"] = self.loss_fn_kwargs
        self.param_dict["optimizer"] = self.optim.__class__.__name__
        self.param_dict["optimizer kwargs"] = self.optim_kwargs
        self.param_dict["momentum"] = self.momentum
        self.param_dict["starting lr"] = self.starting_lr
        # self.param_dict['parameters for augmentation'] = selfmentation_parameters

        if self.lr_scheduler:
            self.param_dict["lr scheduler"] = self.lr_scheduler.__class__.__name__
            self.param_dict["lr scheduler kwargs"] = self.lr_scheduler_kwargs
        if self.grad_scaler:
            self.grad_scaler.__class__.__name__

        # Paths
        self.param_dict["output folder"] = self.outpath
        self.param_dict["log file"] = self.log_file
        self.param_dict["folder with preprocessed files"] = self.folder_with_preprocessed_data

        # Experiment
        self.param_dict["task"] = self.task
        self.param_dict["name"] = self.name
        self.param_dict["random seed"] = self.random_seed
        self.param_dict["command used"] = self.train_command
        save_json(self.param_dict, join(self.outpath, "parameters.json"), sort_keys=False)

    def set_batch_and_patch_sizes(self):
        self.batch_size, self.patch_size = find_optimal_tensor_dims(
            dimensionality=self.model_dimensions,
            num_classes=self.nclasses,
            modalities=self.nmodalities,
            model_name=self.model_name,
            max_patch_size=self.plans["new_mean_size"],
            max_memory_usage_in_gb=self.max_vram,
        )

    def set_train_length(self):
        if self.fast_training:
            self.train_batches_per_epoch = self.fast_train_batches_per_epoch
            self.val_batches_per_epoch = self.fast_val_batches_per_epoch
            self.final_epoch = self.fast_final_epoch

    def set_train_command(self, command):
        self.train_command = command

    def set_outpath(self):
        if not self.outpath:
            if self.starting_lr:
                self.name += "_" + str(self.starting_lr).replace("-", "")

            if self.loss_fn:
                self.name += "_" + str(self.loss_fn)

            if self.momentum:
                self.name += "_" + "m" + str(self.momentum).replace("0.", "")

            if self.fast_training:
                self.name += "_Fast"

            if self.finetune:
                self.name += "_FT"

            self.name += "__" + self.plan_id

            self.outpath = join(
                yucca_models,
                self.task,
                self.model_name,
                self.model_dimensions,
                self.name,
                str(self.folds),
            )

            maybe_mkdir_p(self.outpath)
            self.log(f'{"outpath:":20}', self.outpath, time=False)
        else:
            print("outpath already defined")

    def set_random_seeds(self, seed=None):
        if not self.is_seeded:
            if seed is not None:
                self.random_seed = int(seed)
                self.log("Using the random seed: ", self.random_seed, time=False)
            else:
                self.random_seed = int(mktime(self.initial_time_obj))
                self.log(
                    "No random seed using initial time object: ",
                    self.random_seed,
                    time=False,
                )
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
            self.is_seeded = True
        else:
            print("Random seed already set")
