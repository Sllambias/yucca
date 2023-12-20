import torch
import lightning as L
from dataclasses import dataclass
from typing import Union


@dataclass
class SeedConfig:
    seed: int

    def lm_hparams(self):
        return {"seed": self.seed}


def seed_everything_and_get_seed_config(ckpt_seed: Union[int, None] = None):
    L.seed_everything(seed=ckpt_seed, workers=True)
    seed = torch.initial_seed()

    return SeedConfig(
        seed=seed,
    )
