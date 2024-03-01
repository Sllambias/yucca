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
    A.
        1. Patch based training is false = we train on the full image size of the dataset.
        - This requires the dataset to be preprocessed with a planner that uses "fixed_target_size" such as
        the /planning/resampling/YuccaPlanner_MaxSize.py
    B.
        1. Patch size is set in the manager. This overrides other defaults and inferred patch sizes.
        2. Patch size is set in the ckpt.
        - Relevant for finetuning and inference. This happens if patch size was originally set using the manager.
        In this case we have patch size X from the manager and "default" patch size Y in the plans, where X
        originally overruled Y and thus should do so again.
        3. Patch size is inferred in the plans.
    """
    num_modalities = max(1, plan.get("num_modalities") or len(plan["dataset_properties"]["modalities"]))

    if isinstance(batch_size, str):
        if batch_size == "tiny":
            batch_size = 2
        else:
            raise ValueError(f"Unknown batch size param: {batch_size}")

    # A.1. Get patch size from full image size if we are not doing patch_based_training.
    if not patch_based_training:
        assert plan.get("new_max_size") == plan.get(
            "new_min_size"
        ), "sizes in dataset are not uniform. Non-patch based training only works for datasets with uniform data shapes."
        patch_size = tuple(plan.get("new_max_size"))
        logging.info(f"Getting patch size for non-patch based training")
    else:
        # B.1. Try get patch from manager
        if patch_size is not None:
            logging.info(f"Getting patch size based on manual input of: {patch_size}")
            # Can be three things here: 1. a list/tuple of ints, 2. a list of one int/str or 3. just an int/str
            # First check case 1.
            if isinstance(patch_size, (list, tuple)) and len(patch_size) > 1:
                patch_size = tuple(int(n) for n in patch_size)
            else:
                # Then check case 2 and convert to be identical to case 3.
                if isinstance(patch_size, list) and len(patch_size) == 1:
                    patch_size = patch_size[0]
                # Proceed as if case 3.
                if patch_size in ["max", "min", "mean"]:
                    patch_size = tuple(plan[f"new_{patch_size}_size"])
                elif patch_size == "tiny":
                    patch_size = (32, 32, 32)
                elif patch_size.isdigit():
                    patch_size = (int(patch_size),) * 3
                else:
                    raise ValueError(f"Unknown patch size param: {patch_size}")

        # B.2. Try get patch from ckpt
        elif ckpt_patch_size is not None:
            patch_size = ckpt_patch_size
            logging.info(f"Using patch size found in checkpoint: {ckpt_patch_size}")

        # B.3. Infer patch size from constraints
        else:
            logging.info("Patch size will be infered from hardware constraints")

    # If we have now selected a 3D patch for a 2D model we skip the first dim
    # as we will be extracting patches from that dimension.
    # This also includes if we are not doing patch_based_training.
    if patch_size is not None and model_dimensions == "2D" and len(patch_size) == 3:
        patch_size = patch_size[1:]

    # We ALWAYS run this because even in the case that we have patch_size AND batch_size
    # this function will make sure they're valid by e.g. making the patch size divisible by 16.
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
