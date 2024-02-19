from dataclasses import dataclass
from typing import Tuple, Union, Optional, Literal, Iterable
from yucca.network_architectures.utils.model_memory_estimation import find_optimal_tensor_dims
import logging


@dataclass
class InputDimensionsConfig:
    batch_size: int
    patch_size: Union[Tuple[int, int], Tuple[int, int, int]]
    num_modalities: int

    def lm_hparams(self):
        return {
            "batch_size": self.batch_size,
            "patch_size": self.patch_size,
            "num_modalities": self.num_modalities,
        }


def get_input_dims_config(
    plan: dict,
    model_dimensions: Literal["2D", "3D"],
    model_name: str,
    num_classes: int,
    ckpt_patch_size: Optional[Iterable[int]] = None,
    max_vram: Optional[int] = None,
    batch_size: Union[int, Literal["tiny"]] = None,
    patch_based_training: bool = True,
    patch_size: Union[Literal["max", "min", "mean", "tiny"], Tuple[int, int], Tuple[int, int, int], None] = None,
):
    """
    The priority list for setting patch_size:
    1. Patch based training is false = we train on the full image size of the dataset.
        - This requires the dataset to be preprocessed with a planner that uses "fixed_target_size" such as
        the /planning/resampling/YuccaPlanner_MaxSize.py
    2. Patch size is set in the manager. This overrides other defaults and inferred patch sizes.
    3. Patch size is set in the ckpt.
        - Relevant for finetuning and inference. This happens if patch size was originally set using the manager.
        In this case we have patch size X from the manager and "default" patch size Y in the plans, where X
        originally overruled Y and thus should do so again.
    4. Patch size is inferred in the plans.
    """

    num_modalities = max(1, plan.get("num_modalities") or len(plan["dataset_properties"]["modalities"]))

    # Check patch size priority 1
    if patch_based_training is False:
        assert plan.get("new_max_size") == plan.get(
            "new_min_size"
        ), "sizes in dataset are not uniform. Non-patch based training only works for datasets with uniform data shapes."
        patch_size = tuple(plan.get("new_max_size"))

    # Check patch size priority 2
    if isinstance(patch_size, str):
        if patch_size in ["max", "min", "mean"]:
            # Get patch size from dataset
            patch_size = tuple(plan[f"new_{patch_size}_size"])
        elif patch_size == "tiny":
            patch_size = (32, 32, 32)
        else:
            raise ValueError(f"Unknown patch size param: {patch_size}")

    # Patch size priority 3
    if ckpt_patch_size is not None and patch_size is None:
        patch_size = ckpt_patch_size

    if model_dimensions == "2D" and len(patch_size) == 3:
        # If we have now selected a 3D patch for a 2D model we skip the first dim
        # as we will be extracting patches from that dimension.
        patch_size = patch_size[1:]

    if isinstance(batch_size, str):
        if batch_size == "tiny":
            batch_size = 2
        else:
            raise ValueError(f"Unknown batch size param: {batch_size}")

    # Patch size priority 4
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

    logging.info(f"Using batch size: {batch_size} and patch size: {patch_size}")

    return InputDimensionsConfig(
        batch_size=batch_size,
        patch_size=patch_size,
        num_modalities=num_modalities,
    )
