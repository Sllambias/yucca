import torch
import numpy as np
import yucca
import wandb
from sklearn.model_selection import KFold
from torch import optim
from torch.cuda.amp import GradScaler
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    subfiles,
    save_pickle,
)
from yucca.paths import yucca_preprocessed_data
from yucca.training.trainers.base_manager import base_manager
from yucca.evaluation.confusion_matrix import (
    torch_confusion_matrix_from_logits,
    torch_get_tp_fp_tn_fn,
)
from yucca.evaluation.metrics import dice_per_label
from yucca.loss_and_optim.loss_functions.nnUNet_losses import DiceCE
from yucca.utils.files_and_folders import recursive_find_python_class
from yucca.utils.kwargs import filter_kwargs


class YuccaManager(base_manager):
    """
    This is the barebone functional trainer of the pipeline.

    When training is started it will call self.initialize(). This carries out the following steps:
        - Loads the plans in the folder of the preprocessed training data
        - Sets the required outpaths for saving model weights, training parameters and logs
        - Sets the # epochs and # batches pr. epoch
            - By default 1000 epochs of 250/50 train/val batches pr. epoch
        - Sets the batch size and patch size
          - Patch Size is automatically set based on the dataset.
        - Defines the Data Augmentation scheme
        - Initializes the network
            - Input and output channels are derived from the # modalities and
            # classes stored in the plans file of the dataset.
        - Loads the data
            - If no splits are provided it will automatically split the data.
            - This initializes the data loaders and feeds them to the YuccaAugmenter
        - Initializes the loss function, optimizer, learning rate, scheduler and scaler.
        - Saves all these parameters in the defined outpath
        - Initializes weights & biases
    Then it starts training
        - Using mixed precision
        - Gradients are scaled and clipped.
        - Augmentations are applied using the CPU while the forward - backward pass is handled
        by the GPU to avoid IO bottlenecks
        - On epoch start the start time is recorded and lists for losses etc. are created
        - On epoch finish the end time is recorded and validation results are evaluated and uploaded
        to both WANDB and the local training log. Finally the LR is updated according to any
        scheduler.
        - Model weights are stored every Xth (default 300) epoch AND whenever the model improves its historically
        best Dice on the validation set.
    After the last epoch the final model weights are saved and training is concluded.
    """

    def __init__(
        self,
        model,
        model_dimensions: str,
        task: str,
        folds: str | int,
        planner: str,
        starting_lr: float = None,
        loss_fn: str = None,
        momentum: float = None,
        continue_training: bool = False,
        checkpoint: str = None,
        finetune: bool = False,
        fast_training: bool = False,
    ):
        super().__init__()

        # Trainer specific parameters
        self._DEFAULT_STARTING_LR = 1e-3
        self._DEFAULT_LOSS = DiceCE
        self._DEFAULT_MOMENTUM = 0.9
        self.batch_size_3D = 2
        self.batch_size_2D = 64
        self.train_batches_per_epoch = 250
        self.val_batches_per_epoch = 50
        self.name = self.__class__.__name__
        self.optim = optim.SGD
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR
        self.final_epoch = 1000
        self.p_force_foreground = 0.33
        self.fast_final_epoch = 500
        self.fast_train_batches_per_epoch = 100
        self.fast_val_batches_per_epoch = 20
        self.max_vram = 12

        # REQUIRED:
        # These must be defined in run_training
        self.model_name = model
        self.model_dimensions = model_dimensions
        self.task = task
        self.folds = int(folds)
        self.plan_id = planner
        self.continue_training = continue_training
        self.checkpoint = checkpoint
        self.fast_training = fast_training
        self.finetune = finetune

        # OPTIONALS:
        # These can be defined in run_training
        self.starting_lr = starting_lr
        self.loss_fn = loss_fn
        self.momentum = momentum

        # The default is for the basic UNet.
        self.patch_size_3D = (96, 96, 96)
        self.patch_size_2D = (160, 160)
        if self.model_name == "MultiResUNet":
            self.patch_size_3D = (96, 96, 80)
            self.patch_size_2D = (144, 144)

        # kwargs that we set when params such as LR has been settled (do we use default or is it
        # specified?)
        self.loss_fn_kwargs = {}
        self.optim_kwargs = {}
        self.lr_scheduler_kwargs = {}

    def comprehensive_eval(self, pred, seg):
        if not self.epoch_eval_dict:
            self.epoch_eval_dict = {
                stat: []
                for stat in [
                    "Dice           :",
                    "True Positives :",
                    "False Positives:",
                    "False Negatives:",
                ]
            }

        confusion_matrix = torch_confusion_matrix_from_logits(pred, seg)
        tp, fp, _, fn = torch_get_tp_fp_tn_fn(confusion_matrix, ignore_label=0)
        self.epoch_eval_dict["Dice           :"].append(dice_per_label(tp, fp, _, fn))
        self.epoch_eval_dict["True Positives :"].append(tp)
        self.epoch_eval_dict["False Positives:"].append(fp)
        self.epoch_eval_dict["False Negatives:"].append(fn)

    def initialize(self):
        if not self.is_initialized:
            # Then we load the plans and set modalities and classes etc.
            self.load_plans_from_path(
                join(
                    yucca_preprocessed_data,
                    self.task,
                    self.plan_id,
                    self.plan_id + "_plans.json",
                )
            )

            # First we set the path based on the supplied parameters
            self.set_outpath()

            # Now we can log the plan file, after the outpath has been created
            self.log(f'{"plan file:":20}', self.plans_path, time=False)

            # Now we set up parameters for the data and the data augmentations
            self.set_train_length()
            self.set_batch_and_patch_sizes()
            self.setup_DA()

            # Load self.network based on the
            # dimensions (2D or 3D) and classes (number of output channels)
            self.initialize_network()
            self.log(
                f'{"network:":20}',
                self.model_dimensions,
                self.network.__class__.__name__,
                time=False,
            )

            # Load the data (and if necessary split it)
            # Before loading data, we set random seeds for reproducibility
            self.load_data()

            # Setup optimizer, learning rate, momentum etc.
            self.initialize_loss_optim_lr()

            # Before saving parameters and moving on to training
            # check if we are using pretrained weights/continuing training
            if self.continue_training:
                if self.checkpoint:
                    self.load_checkpoint(self.checkpoint, train=True)
                else:
                    latest = subfiles(self.outpath, suffix="latest.model", join=False)[0]
                    assert latest, "Can not continue training. No latest checkpoint found."

                    self.load_checkpoint(join(self.outpath, latest), train=True)

            # And finally save a json with arguments and parameters
            self.save_parameter_json()
            self.initialize_wandb()

            self.log("initialization done, printing network", "\n", time=False)
            self.log(self.network, time=False)

            self.is_initialized = True
        else:
            print(
                "Network is already initialized. \
                  Calling initialize repeatedly should be avoided."
            )

    def initialize_loss_optim_lr(self):
        # Define the gradscaler
        self.grad_scaler = GradScaler(enabled=True)

        # Defining the loss
        if not self.loss_fn:
            self.loss_fn = self._DEFAULT_LOSS

        elif self.loss_fn:
            self.loss_fn = recursive_find_python_class(
                folder=[join(yucca.__path__[0], "training", "loss_functions")],
                class_name=self.loss_fn,
                current_module="yucca.training.loss_functions",
            )

        self.loss_fn_kwargs = {
            "soft_dice_kwargs": {"apply_softmax": True},  # DCE
            "hierarchical_kwargs": {"rootdict": self.plans["dataset_properties"]["label_hierarchy"]},  # Hierarchical Loss
        }
        self.loss_fn_kwargs = filter_kwargs(self.loss_fn, self.loss_fn_kwargs)
        self.loss_fn = self.loss_fn(**self.loss_fn_kwargs)
        assert isinstance(self.loss_fn, torch.nn.Module), (
            "Loss is not a torch.nn.Module. " "Make sure the correct loss was found (check spelling)"
        )

        self.log(f'{"loss function":20}', self.loss_fn.__class__.__name__, time=False)

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
            "lr": float(self.starting_lr),  # all
            "momentum": float(self.momentum),  # SGD
            "eps": 1e-4,
            "weight_decay": 3e-5,
        }
        self.optim_kwargs = filter_kwargs(self.optim, self.optim_kwargs)
        self.optim = self.optim(self.network.parameters(), **self.optim_kwargs)
        self.log(f'{"optimizer":20}', self.optim.__class__.__name__, time=False)

        # Set kwargs for all schedulers and then filter relevant ones based on scheduler class
        self.lr_scheduler_kwargs = {
            "T_max": self.final_epoch,
            "eta_min": 1e-9,  # Cosine Annealing
        }
        self.lr_scheduler_kwargs = filter_kwargs(self.lr_scheduler, self.lr_scheduler_kwargs)
        self.lr_scheduler = self.lr_scheduler(self.optim, **self.lr_scheduler_kwargs)
        self.log(f'{"LR scheduler":20}', self.lr_scheduler.__class__.__name__, time=False)

    def run_training(self):
        self.initialize()

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            self.log(
                "CUDA NOT AVAILABLE. YOU SHOULD REALLY NOT TRAIN WITHOUT CUDA!",
                time=False,
            )

        wandb.watch(self.network)

        while self.current_epoch < self.final_epoch:
            self.epoch_start()

            self.network.train()
            for _ in range(self.train_batches_per_epoch):
                batch_loss = self.run_batch(next(self.tr_gen))
                self.epoch_tr_loss.append(batch_loss)

            self.network.eval()
            with torch.no_grad():
                for _ in range(self.val_batches_per_epoch):
                    batch_loss = self.run_batch(next(self.val_gen), train=False, comprehensive_eval=True)
                    self.epoch_val_loss.append(batch_loss)

            self.tr_losses.append(np.mean(self.epoch_tr_loss))
            self.val_losses.append(np.mean(self.epoch_val_loss))

            if self.lr_scheduler:
                self.lr_scheduler.step()

            self.epoch_finish()

        if self.current_epoch == self.final_epoch:
            self.save_checkpoint("checkpoint_final.model")
            wandb.finish()

    def setup_DA(self):
        super().setup_DA()
        self.AdditiveNoise_p_per_sample = 0.2
        self.BiasField_p_per_sample = 0.33
        self.Blurring_p_per_sample = 0.2
        self.ElasticDeform_p_per_sample = 0.33
        self.Gamma_p_per_sample = 0.2
        self.GibbsRinging_p_per_sample = 0.2
        self.Mirror_p_per_sample = 0.2
        self.MotionGhosting_p_per_sample = 0.2
        self.MultiplicativeNoise_p_per_sample = 0.2
        self.Rotation_p_per_sample = 0.33
        self.Scale_p_per_sample = 0.33
        self.SimulateLowres_p_per_sample = 0.2

    def split_data(self):
        splits = []

        files = subfiles(self.folder_with_preprocessed_data, join=False, suffix=".npy")
        if not files:
            files = subfiles(self.folder_with_preprocessed_data, join=False, suffix=".npz")
            if files:
                self.log(
                    "Only found compressed (.npz) files. This might increase runtime.",
                    time=False,
                )

        assert files, f"Couldn't find any .npy or .npz files in {self.folder_with_preprocessed_data}"

        files = np.array(files)
        # We set this seed manually as multiple trainers might use this split,
        # And we may not know which individual seed dictated the data splits
        # Therefore for reproducibility this is fixed.

        kf = KFold(n_splits=5, shuffle=True, random_state=52189)
        for train, val in kf.split(files):
            splits.append({"train": list(files[train]), "val": list(files[val])})

        save_pickle(splits, self.splits_file)
