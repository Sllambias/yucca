import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import SimpleProfiler
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from yuccalib.evaluation.loggers import YuccaLogger
from yuccalib.utils.files_and_folders import WritePredictionFromLogits


class CallbackConfigurator:
    def __init__(
        self,
        task: str,
        outpath: str,
        save_dir: str,
        version: int,
        checkpoint_interval: int = 250,
        disable_logging: bool = False,
        prediction_output_dir: str = None,
        profile: bool = False,
        save_softmax: bool = False,
        steps_per_epoch: int = 250,
    ):
        self.checkpoint_interval = checkpoint_interval
        self.disable_logging = disable_logging
        self.outpath = outpath
        self.prediction_output_dir = prediction_output_dir
        self.profile = profile
        self.task = task
        self.save_dir = save_dir
        self.save_softmax = save_softmax
        self.steps_per_epoch = steps_per_epoch
        self.version = version
        self.setup_callbacks()
        self.setup_loggers()
        self.setup_profiler()

    def setup_loggers(self):
        # The YuccaLogger is the barebones logger needed to save hparams.yaml
        # It should generally never be disabled.
        self.loggers = []
        self.loggers.append(
            YuccaLogger(
                disable_logging=self.disable_logging,
                save_dir=self.save_dir,
                name=None,
                version=self.version,
                steps_per_epoch=self.steps_per_epoch,
            )
        )
        if not self.disable_logging:
            self.loggers.append(
                WandbLogger(
                    name=f"version_{self.version}",
                    save_dir=self.outpath,
                    version=str(self.version),
                    project="Yucca",
                    group=self.task,
                    log_model="all",
                )
            )

    def setup_callbacks(self):
        best_ckpt = ModelCheckpoint(
            monitor="val_loss", mode="min", save_top_k=1, filename="best", enable_version_counter=False
        )
        interval_ckpt = ModelCheckpoint(
            every_n_epochs=self.checkpoint_interval, save_top_k=-1, filename="{epoch}", enable_version_counter=False
        )
        latest_ckpt = ModelCheckpoint(
            every_n_epochs=int(np.ceil(self.checkpoint_interval / 10)),
            save_top_k=1,
            filename="last",
            enable_version_counter=False,
        )
        pred_writer = WritePredictionFromLogits(
            output_dir=self.prediction_output_dir, save_softmax=self.save_softmax, write_interval="batch"
        )
        self.callbacks = [best_ckpt, interval_ckpt, latest_ckpt, pred_writer]

    def setup_profiler(self):
        if self.profile:
            self.profiler = SimpleProfiler(dirpath=self.outpath, filename="simple_profile")
        else:
            self.profiler = None
