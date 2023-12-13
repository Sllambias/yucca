from dataclasses import dataclass
from yuccalib.network_architectures.utils.model_memory_estimation import (
    find_optimal_tensor_dims,
)
from typing import Tuple, Union, Optional, Literal


@dataclass
class BatchPatchConfig:
    batch_size: int
    patch_size: Union[Tuple[int, int], Tuple[int, int, int]]

    def lm_hparams(self):
        return {"batch_size": self.batch_size, "patch_size": self.patch_size}


def get_batch_patch_size_cfg(
    plan: dict,
    model_dimensions: Literal["2D", "3D"],
    num_classes: int,
    num_modalities: int,
    model_name: str,
    max_vram: Optional[int] = None,
    batch_size: Optional[int] = None,
    patch_size: Union[Literal["max", "min", "mean", "tiny"], Tuple[int, int], Tuple[int, int, int], None] = None,
):
    # If batch_size or patch_size is not provided, try to infer it from the plan
    if batch_size is None and plan.get("batch_size"):
        batch_size = plan.get("batch_size")
    if patch_size is None and plan.get("patch_size"):
        patch_size = plan.get("patch_size")

    # convert patch_size provided as string to tuple
    if isinstance(patch_size, str):
        if patch_size in ["max", "min", "mean"]:
            # Get patch size from dataset
            patch_size = plan[f"new_{patch_size}_size"]
        elif patch_size == "tiny":
            patch_size = (32, 32, 32)

        if model_dimensions == "2D" and len(patch_size) == 3:
            # If we have now selected a 3D patch for a 2D model we skip the first dim
            # as we will be extracting patches from that dimension.
            patch_size = patch_size[1:]

    if patch_size is None or batch_size is None:
        batch_size, patch_size = find_optimal_tensor_dims(
            fixed_patch_size=patch_size,
            fixed_batch_size=batch_size,
            dimensionality=model_dimensions,
            num_classes=num_classes,
            modalities=num_modalities,
            model_name=model_name,
            max_patch_size=plan["new_mean_size"],
            max_memory_usage_in_gb=max_vram,
        )

    assert isinstance(patch_size, tuple), patch_size
    assert isinstance(batch_size, int), batch_size
    assert batch_size > 0, batch_size
    assert (model_dimensions == "2D" and len(patch_size) == 2) or (model_dimensions == "3D" and len(patch_size) == 3), (
        model_dimensions,
        len(patch_size),
    )

    if plan.get("patch_size") and plan.get("patch_size") != patch_size:
        raise UserWarning(f"Overwriting patch size from plan ({plan.get('patch_size')}) with {patch_size}")
    if plan.get("batch_size") and plan.get("batch_size") != batch_size:
        raise UserWarning(f"Overwriting batch size from plan ({plan.get('batch_size')}) with {batch_size}")

    print(f"Using batch size: {batch_size} and patch size: {patch_size}")

    return BatchPatchConfig(batch_size=batch_size, patch_size=patch_size)
