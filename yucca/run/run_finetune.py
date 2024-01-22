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
    parser.add_argument(
        "--experiment",
        help="A name for the experiment being performed, wiht no spaces.",
        default="finetune",
    )
    # The following can be changed to run training with alternative LR, Loss and/or Momentum ###
    parser.add_argument(
        "--lr",
        help="Should only be used to employ alternative Learning Rate. Format should be scientific notation e.g. 1e-4.",
        default=1e-3,
    )
    parser.add_argument("--loss", help="Should only be used to employ alternative Loss Function", default=None)
    parser.add_argument("--mom", help="Should only be used to employ alternative Momentum.", default=0.9)

    parser.add_argument("--disable_logging", help="disable logging. ", action="store_true", default=False)
    parser.add_argument(
        "--new_version",
        help="Start a new version, instead of continuing from the most recent. ",
        action="store_true",
        default=False,
    )
    parser.add_argument("--profile", help="Enable profiling.", action="store_true", default=False)
    parser.add_argument(
        "--patch_size",
        type=str,
        help="Use your own patch_size. Example: if 32 is provided and the model is 3D we will use patch size (32, 32, 32). Can also be min, max or mean.",
    )
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--train_batches_per_step", type=int, default=250)
    parser.add_argument("--val_batches_per_step", type=int, default=50)
    parser.add_argument("--max_vram", type=int, default=12)

    args = parser.parse_args()

    task = maybe_get_task_from_task_id(args.task)
    checkpoint = args.checkpoint
    model_name = args.m
    dimensions = args.d
    epochs = args.epochs
    experiment = args.experiment
    manager_name = args.man
    momentum = args.mom
    lr = args.lr
    log = not args.disable_logging
    loss = args.loss
    new_version = args.new_version
    patch_size = args.patch_size
    planner = args.pl
    profile = args.profile
    split_idx = int(args.f)
    if patch_size is not None:
        if patch_size not in ["mean", "max", "min"]:
            patch_size = (int(patch_size),) * 3 if dimensions == "3D" else (int(patch_size),) * 2

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
        enable_logging=log,
        experiment=experiment,
        loss=loss,
        learning_rate=lr,
        max_epochs=args.epochs,
        max_vram=args.max_vram,
        model_dimensions=dimensions,
        model_name=model_name,
        momentum=momentum,
        num_workers=8,
        patch_size=patch_size,
        planner=planner,
        precision=args.precision,
        profile=profile,
        split_idx=split_idx,
        step_logging=False,
        task=task,
        train_batches_per_step=args.train_batches_per_step,
        val_batches_per_step=args.val_batches_per_step,
        **kwargs,
    )
    manager.run_finetuning()


if __name__ == "__main__":
    main()
