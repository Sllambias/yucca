import argparse
import yucca
from yucca.pipeline.task_conversion.utils import get_task_from_task_id
from yucca.functional.utils.files_and_folders import recursive_find_python_class
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
        "-d",
        "--dimension",
        help="Dimensionality of the Model. Can be 3D or 2D. "
        "Defaults to 3D. Note that this will always be 2D if ensemble is enabled.",
        default="3D",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Model Architecture. Should be one of MultiResUNet or UNet"
        " Note that this is case sensitive. "
        "Defaults to the standard UNet.",
        default="UNet",
    )
    parser.add_argument(
        "-man",
        "--manager",
        help="Manager Class to be used. " "Defaults to the basic YuccaManager",
        default="YuccaManager",
    )
    parser.add_argument(
        "-pl",
        "--plan",
        help="Plan ID to be used. "
        "This specifies which plan and preprocessed data to use for training "
        "on the given task. Defaults to the YuccaPlanner folder",
        default="YuccaPlanner",
    )

    # Optionals that can be changed experimentally. For long term solutions these should be specified by a unique Manager.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size to be used for training. Overrides the batch size specified in the plan.",
    )
    parser.add_argument("--disable_logging", help="disable logging.", action="store_true", default=False)
    parser.add_argument(
        "-ds", "--deep_supervision", help="Used to enable deep supervision", default=False, action="store_true"
    )
    parser.add_argument(
        "--epochs", help="Used to specify the number of epochs for training. Default is 1000", type=int, default=1000
    )
    parser.add_argument("--experiment", help="A name for the experiment being performed, with no spaces.", default="default")
    parser.add_argument("--loss", help="Should only be used to employ alternative Loss Function", default=None)
    parser.add_argument(
        "--lr",
        help="Should only be used to employ alternative Learning Rate. Format should be scientific notation e.g. 1e-4.",
        default=1e-3,
    )
    parser.add_argument("--max_vram", type=int, default=12)
    parser.add_argument("--mom", help="Should only be used to employ alternative Momentum.", default=0.9)
    parser.add_argument(
        "--new_version",
        help="Start a new version, instead of continuing from the most recent. ",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Num workers used in the DataLoaders. By default this will be inferred from the number of available CPUs-1",
        default=None,
    )
    parser.add_argument(
        "--patch_size",
        nargs="+",
        type=str,
        help="Use your own patch_size. Example: if 32 is provided and the model is 3D we will use patch size (32, 32, 32). This patch size can be set manually by passing 32 32 32 as arguments. The argument can also be min, max or mean.",
        default=None,
    )
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--profile", help="Enable profiling.", action="store_true", default=False)
    parser.add_argument("--split_idx", type=int, help="idx of splits to use for training.", default=0)
    parser.add_argument(
        "--split_data_method", help="Specify splitting method. Either kfold, simple_train_val_split", default="kfold"
    )
    parser.add_argument(
        "--split_data_param",
        help="Specify the parameter for the selected split method. For KFold use an int, for simple_split use a float between 0.0-1.0.",
        default=5,
    )
    parser.add_argument("--train_batches_per_step", type=int, default=250)
    parser.add_argument("--val_batches_per_step", type=int, default=50)
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()

    # Required
    task = get_task_from_task_id(args.task, stage="preprocessed")

    # Optionals (frequently changed)
    dimensions = args.dimension
    model_name = args.model
    manager_name = args.manager
    planner = args.plan

    # Optionals (occasionally changed)
    log = not args.disable_logging
    batch_size = args.batch_size
    deep_supervision = args.deep_supervision
    epochs = args.epochs
    experiment = args.experiment
    fast_dev_run = args.fast_dev_run
    loss = args.loss
    lr = args.lr
    max_vram = args.max_vram
    momentum = args.mom
    new_version = args.new_version
    num_workers = args.num_workers
    patch_size = args.patch_size
    precision = args.precision
    profile = args.profile
    split_idx = args.split_idx
    split_data_method = args.split_data_method
    split_data_param = args.split_data_param
    train_batches_per_step = args.train_batches_per_step
    val_batches_per_step = args.val_batches_per_step

    kwargs = {"fast_dev_run": fast_dev_run}

    assert model_name in [
        "MedNeXt",
        "mednext_s3",
        "mednext_m3",
        "MultiResUNet",
        "UNet",
        "UNetR",
        "UNetRE",
        "UNetS",
        "UXNet",
        "ResNet50",
        "TinyUNet",
    ], f"{model_name} is an invalid model name. This is case sensitive."

    print(f"{'Using manager: ':25} {manager_name}")
    manager = recursive_find_python_class(
        folder=[join(yucca.__path__[0], "pipeline", "managers")],
        class_name=manager_name,
        current_module="yucca.pipeline.managers",
    )

    manager = manager(
        ckpt_path=None,
        continue_from_most_recent=not new_version,
        deep_supervision=deep_supervision,
        enable_logging=log,
        loss=loss,
        learning_rate=lr,
        max_epochs=epochs,
        max_vram=max_vram,
        model_dimensions=dimensions,
        model_name=model_name,
        momentum=momentum,
        num_workers=num_workers,
        patch_size=patch_size,
        planner=planner,
        precision=precision,
        profile=profile,
        split_idx=split_idx,
        split_data_method=split_data_method,
        split_data_param=split_data_param,
        step_logging=False,
        task=task,
        batch_size=batch_size,
        experiment=experiment,
        train_batches_per_step=train_batches_per_step,
        val_batches_per_step=val_batches_per_step,
        **kwargs,
    )
    manager.run_training()


if __name__ == "__main__":
    main()
