import numpy as np
from typing import Union
from dataclasses import dataclass
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import SimpleProfiler
from pytorch_lightning.loggers import WandbLogger
from yucca.evaluation.loggers import YuccaLogger
from yucca.utils.saving import WritePredictionFromLogits
from lightning.pytorch.profilers.profiler import Profiler


@dataclass
class CallbackConfig:
    callbacks: list
    loggers: list
    profiler: Profiler
    wandb_id: str

    def lm_hparams(self):
        return {"wandb_id": self.wandb_id}


def get_callback_config(
    task: str,
    save_dir: str,
    version_dir: str,
    version: int,
    interval_ckpt_epochs: int = 250,
    latest_ckpt_epochs: int = 25,
    disable_logging: bool = False,
    prediction_output_dir: str = None,
    profile: bool = False,
    save_softmax: bool = False,
    steps_per_epoch: int = 250,
    project: str = "Yucca",
    log_model: str = "all",
    write_predictions: bool = True,
    store_best_ckpt: bool = True,
):
    callbacks = get_callbacks(
        interval_ckpt_epochs, latest_ckpt_epochs, prediction_output_dir, save_softmax, write_predictions, store_best_ckpt
    )
    loggers = get_loggers(task, save_dir, version_dir, version, disable_logging, steps_per_epoch, project, log_model)
    wandb_id = get_wandb_id(loggers, disable_logging)
    profiler = get_profiler(profile, save_dir)

    return CallbackConfig(callbacks=callbacks, loggers=loggers, profiler=profiler, wandb_id=wandb_id)


def get_loggers(
    task: str,
    save_dir: str,
    version_dir: str,
    version: Union[int, str],
    disable_logging: bool = False,
    steps_per_epoch: int = 250,
    project: str = "Yucca",
    log_model: str = "all",
):
    # The YuccaLogger is the barebones logger needed to save hparams.yaml
    # It should generally never be disabled.
    if isinstance(version, str):
        version = int(version)

    loggers = [
        YuccaLogger(
            disable_logging=disable_logging,
            save_dir=save_dir,
            name=None,
            version=version,
            steps_per_epoch=steps_per_epoch,
        )
    ]
    if not disable_logging:
        loggers.append(
            WandbLogger(
                name=f"version_{version}",
                save_dir=version_dir,
                version=str(version),
                project=project,
                group=task,
                log_model=log_model,
            )
        )
    return loggers


def get_callbacks(
    interval_ckpt_epochs: int,
    latest_ckpt_epochs: int,
    prediction_output_dir: str = None,
    save_softmax: bool = False,
    write_predictions: bool = True,
    store_best_ckpt: bool = True,
):
    interval_ckpt = ModelCheckpoint(
        every_n_epochs=interval_ckpt_epochs, save_top_k=-1, filename="{epoch}", enable_version_counter=False
    )
    latest_ckpt = ModelCheckpoint(
        every_n_epochs=latest_ckpt_epochs,
        save_top_k=1,
        filename="last",
        enable_version_counter=False,
    )

    callbacks = [interval_ckpt, latest_ckpt]

    if store_best_ckpt:
        best_ckpt = ModelCheckpoint(
            monitor="val_loss", mode="min", save_top_k=1, filename="best", enable_version_counter=False
        )
        callbacks.append(best_ckpt)

    if write_predictions:
        pred_writer = WritePredictionFromLogits(
            output_dir=prediction_output_dir, save_softmax=save_softmax, write_interval="batch"
        )
        callbacks.append(pred_writer)

    return callbacks


def get_profiler(profile, outpath):
    if profile:
        return SimpleProfiler(dirpath=outpath, filename="simple_profile")
    else:
        return None


def get_wandb_id(loggers, disable_logging):
    if not disable_logging:
        return loggers[-1].experiment.id
    else:
        return None
