import yucca
import torch
from batchgenerators.utilities.file_and_folder_operations import join, isdir, subdirs, maybe_mkdir_p, isfile, load_json
from dataclasses import dataclass
from typing import Union
from yucca.paths import yucca_models, yucca_preprocessed_data
from yucca.preprocessing.UnsupervisedPreprocessor import UnsupervisedPreprocessor
from yucca.preprocessing.ClassificationPreprocessor import ClassificationPreprocessor
from yucca.training.configuration.configure_paths import PathConfig
from yucca.utils.dict import without_keys
from yucca.utils.files_and_folders import recursive_find_python_class


@dataclass
class CkptConfig:
    """
    This class serves to preserve information between runs and should thus be extended to include
    information that must not be overwritten by each run.

    The LightningModule will call save_hyperparameters each time is it instantiated. This includes:
    starting training the first time, starting finetuning, continuing an interrupted training,
    starting inference. And each time the hyperparameters will be overwritten.

    ckpt_path will preserve the path to the original weights - this will by default also be saved
    when finetuning is started, but if finetuning is stopped and then started again the path is wiped.

    ckpt_seed ensures that we can continue from the same seed - this is mostly relevant for continuing
    interrupted trainings, but will also affect finetuning.

    ckpt_plans allows us to load plans from the checkpoint - this is used in inference where we may
    not be able to find a preprocessed folder with a plans.json.

    ckpt_wandb_id allows us to group runs by their "parent" wandb_id - this is useful for visualizing
    groups of finetuning runs by their source model/run.
    NB: THIS WILL NOT BE USED TO RESUME LOGGER STATES. Resuming loggers is currently not supported.
        for active discussion see: https://github.com/Lightning-AI/pytorch-lightning/issues/5342
        for workaround see: https://github.com/Lightning-AI/pytorch-lightning/issues/13524
    """

    ckpt_path: Union[str, None]
    ckpt_seed: Union[int, None]
    ckpt_plans: Union[dict, None]
    ckpt_version_dir: Union[str, None]
    ckpt_wandb_id: Union[str, None]

    def lm_hparams(self, without: [] = []):
        hparams = {
            "ckpt_path": self.ckpt_path,
            "ckpt_seed": self.ckpt_seed,
            "ckpt_plans": self.ckpt_plans,
            "ckpt_version_dir": self.ckpt_version_dir,
            "ckpt_wandb_id": self.ckpt_wandb_id,
        }
        return without_keys(hparams, without)


def get_checkpoint_config(path_config: PathConfig, continue_from_most_recent: bool, ckpt_path: Union[str, None] = None):
    ckpt_path = find_checkpoint_path(
        ckpt_path=ckpt_path,
        continue_from_most_recent=continue_from_most_recent,
        version=path_config.version,
        version_dir=path_config.version_dir,
    )

    # If we did not find a checkpoint we just return an empty ckpt_config
    if ckpt_path is None:
        return CkptConfig(ckpt_path=None, ckpt_seed=None, ckpt_plans=None, ckpt_version_dir=None, ckpt_wandb_id=None)

    checkpoint = torch.load(ckpt_path, map_location="cpu")["hyper_parameters"]["config"]
    plans, seed, version_dir, wandb_id = get_checkpoint_params(checkpoint)

    return CkptConfig(
        ckpt_path=ckpt_path,
        ckpt_seed=seed,
        ckpt_plans=plans,
        ckpt_version_dir=version_dir,
        ckpt_wandb_id=wandb_id,
    )


def find_checkpoint_path(ckpt_path: Union[str, None], continue_from_most_recent: bool, version: int, version_dir: str):
    if ckpt_path:
        assert isfile(ckpt_path)
        print(f"Using ckpt file: {ckpt_path}")
        return ckpt_path
    elif version is not None and continue_from_most_recent and isfile(join(version_dir, "checkpoints", "last.ckpt")):
        print("Using last checkpoint and continuing training")
        return join(version_dir, "checkpoints", "last.ckpt")
    else:
        return None


def get_checkpoint_params(checkpoint: dict):
    plans = checkpoint.get("plans") if checkpoint.get("plans") != "null" else None
    seed = checkpoint.get("seed") if checkpoint.get("seed") != "null" else None
    version_dir = checkpoint.get("version_dir") if checkpoint.get("version_dir") != "null" else None
    wandb_id = checkpoint.get("wandb_id") if checkpoint.get("wandb_id") != "null" else None
    return plans, seed, version_dir, wandb_id
