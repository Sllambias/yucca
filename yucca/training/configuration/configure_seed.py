import torch
import lightning as L
import datetime
from dataclasses import dataclass
from typing import Union


@dataclass
class SeedConfig:
    seed: int

    def lm_hparams(self):
        return {"seed": self.seed}


def seed_everything_and_get_seed_config(manual_seed: Union[int, None] = None, ckpt_seed: Union[int, None] = None):
    # Priority of seeds.
    # 1. If manual seed is specified we use that
    # 2. If seed is found in checkpoint continue with that (e.g. continued training or finetuning)
    # 3. If no seed is found we generate one from the datetime object
    if manual_seed is not None:
        seed = manual_seed
    elif ckpt_seed is not None:
        seed = ckpt_seed
    else:
        dt = datetime.datetime.now()
        seq = int(dt.strftime("%m%d%H%M%S"))
        seed = seq

    L.seed_everything(seed=seed, workers=True)
    seed = torch.initial_seed()

    return SeedConfig(
        seed=seed,
    )
