from lightning.pytorch.loggers.logger import Logger
from typing import Union
from dataclasses import dataclass
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.loggers import WandbLogger
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
    model_name: str,
    save_dir: str,
    task: str,
    version_dir: str,
    version: int,
    ckpt_version_dir: Union[str, None] = None,
    ckpt_wandb_id: Union[str, None] = None,
    enable_logging: bool = True,
    interval_ckpt_epochs: int = 250,
    latest_ckpt_epochs: int = 25,
    log_lr: bool = True,
    log_model: str = "all",
    prediction_output_dir: str = None,
    profile: bool = False,
    project: str = "Yucca",
    save_softmax: bool = False,
    steps_per_epoch: int = 250,
    store_best_ckpt: bool = True,
    write_predictions: bool = True,
    run_name: str = None,
    run_description: str = None,
):
    callbacks = get_callbacks(
        interval_ckpt_epochs,
        latest_ckpt_epochs,
        prediction_output_dir,
        save_softmax,
        write_predictions,
        store_best_ckpt,
        log_lr,
    )
    loggers = get_loggers(
        ckpt_version_dir=ckpt_version_dir,
        ckpt_wandb_id=ckpt_wandb_id,
        task=task,
        model_name=model_name,
        save_dir=save_dir,
        version_dir=version_dir,
        version=version,
        enable_logging=enable_logging,
        steps_per_epoch=steps_per_epoch,
        project=project,
        log_model=log_model,
        run_name=run_name,
        run_description=run_description,
    )
    wandb_id = get_wandb_id(loggers, enable_logging)
    profiler = get_profiler(profile, save_dir)

    return CallbackConfig(callbacks=callbacks, loggers=loggers, profiler=profiler, wandb_id=wandb_id)


def get_loggers(
    ckpt_version_dir: Union[str, None],
    ckpt_wandb_id: Union[str, None],
    enable_logging: bool,
    log_model: str,
    model_name: str,
    project: str,
    save_dir: str,
    steps_per_epoch: int,
    task: str,
    version_dir: str,
    version: Union[int, str],
    run_name: str,
    run_description: str,
):
    # The YuccaLogger is the barebones logger needed to save hparams.yaml
    # It should generally never be disabled.
    if isinstance(version, str):
        version = int(version)

    loggers = [
        YuccaLogger(
            disable_logging=not enable_logging,
            save_dir=save_dir,
            name=None,
            version=version,
            steps_per_epoch=steps_per_epoch,
        )
    ]
    if enable_logging:
        use_ckpt_id = should_use_ckpt_wandb_id(ckpt_version_dir, ckpt_wandb_id, version_dir)
        loggers.append(
            WandbLogger(
                name=run_name or f"version_{version}",
                notes=run_description,
                save_dir=version_dir,
                project=project,
                group=task,
                log_model=log_model,
                version=ckpt_wandb_id if use_ckpt_id else None,
            )
        )

    return loggers


def get_callbacks(
    interval_ckpt_every_n_epochs: int,
    last_ckpt_every_n_epochs: int,
    prediction_output_dir: str,
    save_softmax: bool,
    write_predictions: bool,
    store_best_ckpt: bool,
    log_lr: bool,
):
    interval_ckpt = ModelCheckpoint(
        every_n_epochs=interval_ckpt_every_n_epochs, save_top_k=-1, filename="{epoch}", enable_version_counter=False
    )
    latest_ckpt = ModelCheckpoint(
        every_n_epochs=last_ckpt_every_n_epochs,
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

    if log_lr:
        lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=True)
        callbacks.append(lr_monitor)

    return callbacks


def get_profiler(profile: bool, outpath: str):
    if profile:
        return SimpleProfiler(dirpath=outpath, filename="simple_profile")
    return None


def get_wandb_id(loggers: list[Logger], enable_logging: bool):
    if enable_logging:
        wandb_logger = loggers[-1]
        assert isinstance(wandb_logger, WandbLogger)
        return wandb_logger.experiment.id
    return None


def should_use_ckpt_wandb_id(ckpt_version_dir, ckpt_wandb_id, version_dir):
    # If no wandb_id was found we can not (and should not try) reuse it.
    if ckpt_wandb_id is None:
        return False
    # If it exists and our current output directory INCLUDING THE CURRENT VERSION is equal
    # to the previous output directory we can safely assume we're continuing an
    # interrupted run.
    if ckpt_version_dir == version_dir:
        return True
