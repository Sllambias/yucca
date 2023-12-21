import argparse
import yucca
from yucca.paths import yucca_models
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
    parser.add_argument("-chk", "--checkpoint", help="Path to the checkpoint from where the weights should be restored. ")

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
    parser.add_argument(
        "-f",
        help="Fold to use for training. Unless manually assigned, "
        "folds [0,1,2,3,4] will be created automatically. "
        "Defaults to training on fold 0",
        default=0,
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

    args = parser.parse_args()

    task = maybe_get_task_from_task_id(args.task)
    checkpoint = args.checkpoint
    model_name = args.m
    dimensions = args.d
    epochs = args.epochs
    manager_name = args.man
    split_idx = int(args.f)
    lr = args.lr
    log = not args.disable_logging
    loss = args.loss
    momentum = args.mom
    new_version = args.new_version
    planner = args.pl
    profile = args.profile

    kwargs = {}

    if lr:
        assert "e" in lr, f"Learning Rate should be in scientific notation e.g. 1e-4, but is {lr}"

    manager = recursive_find_python_class(
        folder=[join(yucca.__path__[0], "training", "managers")],
        class_name=manager_name,
        current_module="yucca.training.managers",
    )
    # checkpoint = join(
    #    yucca_models,
    #    source_task,
    #    model + "__" + dimensions,
    #    manager_name + "__" + planner,
    #    f"fold_{str(folds)}",
    #    f"version_{str(version)}",
    #    "checkpoints",
    #    f"{checkpoint}.ckpt",
    # )

    manager = manager(
        ckpt_path=checkpoint,
        continue_from_most_recent=not new_version,
        deep_supervision=False,
        disable_logging=log,
        split_idx=split_idx,
        loss=loss,
        model_dimensions=dimensions,
        model_name=model_name,
        num_workers=8,
        planner=planner,
        precision="16-mixed",
        profile=profile,
        step_logging=False,
        task=task,
        **kwargs,
    )
    manager.run_finetuning()


if __name__ == "__main__":
    main()
