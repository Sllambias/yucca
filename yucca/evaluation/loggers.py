import sys
import os
import logging
from argparse import Namespace
from lightning.pytorch.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.core.saving import save_hparams_to_yaml
from lightning_fabric.utilities.logger import _convert_params
from time import localtime, strftime, time
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p,
    isdir,
)
from typing import Any, Dict, Optional, Union


class YuccaLogger(Logger):
    def __init__(
        self,
        disable_logging: bool = False,
        save_dir: str = "./",
        name: str = "",
        steps_per_epoch: int = None,
        version: Optional[Union[int, str]] = None,
    ):
        super().__init__()
        self._name = name or ""
        self._root_dir = save_dir
        self._version = version
        self.disable_logging = disable_logging
        self.steps_per_epoch = steps_per_epoch

        # Default params
        self.epoch_start_time = time()
        self.log_file = None
        self.previous_epoch = 0
        self.hparams: Dict[str, Any] = {}
        self.NAME_HPARAMS_FILE = "hparams.yaml"

        if self.disable_logging is False:
            if self.log_file is None:
                self.create_logfile()
            self.duplicate_console_out_to_log_file(self.log_file)

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def log_dir(self):
        log_dir = self.root_dir
        if self.name is not None:
            log_dir = join(log_dir, self.name)
        if self.version is not None:
            version = self.version if isinstance(self.version, str) else f"version_{self.version}"
            log_dir = join(log_dir, version)
        if not isdir(log_dir):
            maybe_mkdir_p(log_dir)
        return log_dir

    @rank_zero_only
    def create_logfile(self):
        maybe_mkdir_p(self.log_dir)
        self.log_file = join(
            self.log_dir,
            "training_log.txt",
        )
        with open(self.log_file, "w") as f:
            f.write("Starting model training")
            logging.info("Starting model training \n" f'{"log file:":20} {self.log_file} \n')
            f.write("\n")
            f.write(f'{"log file:":20} {self.log_file}')
            f.write("\n")

    @rank_zero_only
    def duplicate_console_out_to_log_file(self, log_file):
        # Add the log_file as a duplicate handler of lightning outputs
        logging.getLogger("lightning.pytorch").addHandler(logging.FileHandler(log_file))
        # Add the log_file as a duplicate handler of lightning outputs
        logging.getLogger().addHandler(logging.FileHandler(log_file))

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:  # type: ignore[override]
        params = _convert_params(params)
        self.log_hparams(params)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        if self.disable_logging:
            return
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        t = strftime("%Y_%m_%d_%H_%M_%S", localtime())
        with open(self.log_file, "a+") as f:
            current_epoch = (step + 1) // self.steps_per_epoch
            if current_epoch != self.previous_epoch:
                epoch_end_time = time()
                f.write("\n")
                f.write("\n")
                print("\n")
                f.write(f"{t} {'Current Epoch:':20} {current_epoch} \n")
                f.write(f"{t} {'Epoch Time:':20} {epoch_end_time-self.epoch_start_time} \n")
                print(f"{t} {'Current Epoch:':20} {current_epoch}")
                print(f"{t} {'Epoch Time:':20} {epoch_end_time-self.epoch_start_time}")
                self.previous_epoch = current_epoch
                self.epoch_start_time = epoch_end_time
            for key in metrics:
                if key == "epoch":
                    continue
                f.write(f"{t} {key+':':20} {metrics[key]} \n")
                print(f"{t} {key+':':20} {metrics[key]}")
        sys.stdout.flush()

    @rank_zero_only
    def log_hparams(self, params: Dict[str, Any]) -> None:
        """Record hparams."""
        self.hparams.update(params)

    @rank_zero_only
    def save(self) -> None:
        """Save recorded hparams into yaml."""
        hparams_file = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)
        save_hparams_to_yaml(hparams_file, self.hparams)

    @rank_zero_only
    def finalize(self, _status) -> None:
        # When using multiprocessing, finalize() should be a no-op on the main process, as no experiment has been
        # initialized there
        self.save()
