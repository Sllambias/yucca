import argparse
import yucca
from yucca.utils.task_ids import maybe_get_task_from_task_id
from yucca.utils.files_and_folders import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join


def main():
    parser = argparse.ArgumentParser()

    # Required Arguments #
    parser.add_argument(
        "-t",
        "--task",
        help="Name of the task used for training. "
        "The data should already be preprocessed using yucca_preprocess"
        "Argument should be of format: TaskXXX_MYTASK",
    )

    # Optional arguments with default values #
    parser.add_argument(
        "-m",
        help="Model Architecture. Should be one of MultiResUNet or UNet"
        " Note that this is case sensitive. "
        "Defaults to the standard UNet.",
        default="UNet",
    )
    parser.add_argument(
        "-d",
        help="Dimensionality of the Model. Can be 3D or 2D. "
        "Defaults to 3D. Note that this will always be 2D if ensemble is enabled.",
        default="3D",
    )
    parser.add_argument(
        "-man",
        help="Manager Class to be used. " "Defaults to the basic YuccaManager",
        default="YuccaManager",
    )
    parser.add_argument(
        "-pl",
        help="Plan ID to be used. "
        "This specifies which plan and preprocessed data to use for training "
        "on the given task. Defaults to the YuccaPlanne folder",
        default="YuccaPlanner",
    )

    parser.add_argument("--epochs", help="Used to specify the number of epochs for training. Default is 1000")
    # The following can be changed to run training with alternative LR, Loss and/or Momentum ###
    parser.add_argument(
        "--lr",
        help="Should only be used to employ alternative Learning Rate. Format should be scientific notation e.g. 1e-4.",
        default=None,
    )
    parser.add_argument("--loss", help="Should only be used to employ alternative Loss Function", default=None)
    parser.add_argument("--mom", help="Should only be used to employ alternative Momentum.", default=None)
    parser.add_argument("--ds", help="Used to enable deep supervision", default=False, action="store_true")
    parser.add_argument(
        "--disable_logging",
        help="disable logging. ",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--new_version",
        help="Start a new version, instead of continuing from the most recent. ",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--profile",
        help="Enable profiling.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--experiment",
        help="A name for the experiment being performed, wiht no spaces.",
        default="default",
    )

    parser.add_argument(
        "--patch_size",
        type=str,
        help="Use your own patch_size. Example: if 32 is provided and the model is 3D we will use patch size (32, 32, 32). Can also be min, max or mean.",
        default=None,
    )

    # Split configs
    parser.add_argument(
        "-p",
        type=float,
        help="Use a simple train/val split where `p` is the fraction of items used for the val split.",
        default=None,
    )
    parser.add_argument("-k", type=int, help="Use kfold split where `k` is amount of folds.", default=None)
    parser.add_argument("-f", "--split_idx", type=int, help="idx of splits to use for training.", default=0)

    parser.add_argument("--precision", type=str, default="bf16-mixed")

    args = parser.parse_args()

    task = maybe_get_task_from_task_id(args.task)
    model_name = args.m
    dimensions = args.d
    deep_supervision = args.ds
    epochs = args.epochs
    manager_name = args.man
    lr = args.lr
    log = not args.disable_logging
    loss = args.loss
    momentum = args.mom
    patch_size = args.patch_size
    new_version = args.new_version
    planner = args.pl
    profile = args.profile
    experiment = args.experiment

    split_idx = args.split_idx
    k = args.k
    p = args.p

    if k is None and p is None:
        k = 5

    if patch_size is not None:
        if patch_size not in ["mean", "max", "min"]:
            patch_size = (int(patch_size),) * 3 if dimensions == "3D" else (int(patch_size),) * 2

    # checkpoint = args.chk
    kwargs = {}

    if epochs:
        kwargs["max_epochs"] = int(epochs)

    assert model_name in [
        "MedNeXt",
        "MultiResUNet",
        "UNet",
        "UNetR",
        "UNetRE",
        "UXNet",
        "ResNet50",
        "TinyUNet",
    ], f"{model_name} is an invalid model name. This is case sensitive."

    print(f"{'Using manager: ':25} {manager_name}")
    manager = recursive_find_python_class(
        folder=[join(yucca.__path__[0], "training", "managers")],
        class_name=manager_name,
        current_module="yucca.training.managers",
    )

    manager = manager(
        ckpt_path=None,
        continue_from_most_recent=not new_version,
        deep_supervision=deep_supervision,
        enable_logging=log,
        split_idx=split_idx,
        k=k,
        p=p,
        loss=loss,
        model_dimensions=dimensions,
        model_name=model_name,
        num_workers=8,
        planner=planner,
        precision=args.precision,
        profile=profile,
        step_logging=False,
        task=task,
        experiment=experiment,
        **kwargs,
    )
    manager.run_training()


if __name__ == "__main__":
    main()
