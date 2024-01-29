import argparse
import yucca
from yucca.utils.task_ids import maybe_get_task_from_task_id
from yucca.paths import yucca_raw_data, yucca_results, yucca_models
from yucca.evaluation.YuccaEvaluator import YuccaEvaluator
from yucca.training.managers.YuccaManager import YuccaManager
from yucca.utils.files_and_folders import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    isfile,
    maybe_mkdir_p,
    isdir,
    subdirs,
)
from warnings import filterwarnings

filterwarnings("ignore")


def main():
    from warnings import filterwarnings

    filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    # Required Arguments
    parser.add_argument(
        "-s",
        help="Name of the source task i.e. what the model is trained on. " "Should be of format: TaskXXX_MYTASK",
        required=True,
    )
    parser.add_argument(
        "-t",
        help="Name of the target task i.e. the data to be predicted. " "Should be of format: TaskXXX_MYTASK",
        required=True,
    )

    # Optionals (frequently changed)
    parser.add_argument(
        "-chk",
        "--checkpoint",
        help="Checkpoint to use for inference. Defaults to model_best.",
        default="best",
    )
    parser.add_argument("-d", "--dimensions", help="2D or 3D model. Defaults to 3D.", default="3D")

    parser.add_argument(
        "-f",
        "--fold",
        help="Select the fold that was used to train the model desired for inference. "
        "Defaults to looking for a model trained on fold 0.",
        default="0",
    )
    parser.add_argument("-m", "--model", help="Model Architecture. Defaults to UNet.", default="UNet")
    parser.add_argument(
        "-man",
        "--manager",
        help="Full name of Trainer Class. \n" "e.g. 'YuccaTrainer_DCE' or 'YuccaTrainer'. Defaults to YuccaTrainer.",
        default="YuccaManager",
    )
    parser.add_argument("-pl", "--planner", help="Planner. Defaults to YuccaPlanner", default="YuccaPlanner")
    parser.add_argument(
        "-v",
        "--version",
        help="Version to use for inference. Defaults to the newest version.",
        default=None,
    )
    # Optionals (occasionally changed)
    parser.add_argument(
        "--experiment",
        help="A name for the experiment being performed, with no spaces.",
        default="default",
    )

    parser.add_argument(
        "--disable_tta",
        help="Used to disable test-time augmentations (mirroring)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--no_eval",
        help="Disable evaluation and creation of metrics file (result.json)",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--no_wandb",
        help="Disable logging of evaluation results to wandb",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--no_sliding_window",
        help="Disable sliding window prediction and instead use fixed patch/input size",
        default=False,
        action="store_true",
        required=False,
    )
    # parser.add_argument(
    #    "--overwrite",
    #    default=False,
    #    action="store_true",
    #    required=False,
    #    help="Overwrite existing predictions",
    # )
    parser.add_argument(
        "--predict_train",
        default=False,
        action="store_true",
        required=False,
        help="Predict on the training set. Useful for debugging.",
    )
    parser.add_argument(
        "--profile",
        help="Used to enable inference profiling",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--save_softmax",
        default=False,
        action="store_true",
        required=False,
        help="Save softmax outputs. Required for softmax fusion.",
    )
    parser.add_argument(
        "--task_type",
        default="segmentation",
        type=str,
        required=False,
        help="Defaults to segmentation. Set to 'classification' for classification tasks.",
    )

    args = parser.parse_args()

    # Required
    source_task = maybe_get_task_from_task_id(args.s)
    target_task = maybe_get_task_from_task_id(args.t)

    # Optionals (frequently changed)
    checkpoint = args.checkpoint
    dimensions = args.dimensions
    manager_name = args.manager
    model = args.model
    planner = args.planner
    profile = args.profile
    split_idx = int(args.fold)
    version = args.version
    task_type = args.task_type

    # Optionals (occasionally changed)
    experiment = args.experiment
    disable_tta = args.disable_tta
    no_eval = args.no_eval
    # overwrite = args.overwrite
    predict_train = args.predict_train
    save_softmax = args.save_softmax
    use_wandb = not args.no_wandb

    batch_size = 1  # batch size is fixed to 1 for inference, as write_predictions is implemented on_batch_end

    path_to_versions = join(
        yucca_models, source_task, model + "__" + dimensions, manager_name + "__" + planner, experiment, f"fold_{split_idx}"
    )
    if version is None:
        versions = [int(i.split("_")[-1]) for i in subdirs(path_to_versions, join=False)]
        version = str(max(versions))

    modelfile = join(
        yucca_models,
        source_task,
        model + "__" + dimensions,
        manager_name + "__" + planner,
        experiment,
        f"fold_{split_idx}",
        f"version_{version}",
        "checkpoints",
        checkpoint + ".ckpt",
    )

    assert isfile(modelfile), "Can't find .cpkt file with trained model weights. " f"Should be located at: {modelfile}"
    print(f"######################################################################## \n" f"{'Using model: ':25} {modelfile}")

    manager = recursive_find_python_class(
        folder=[join(yucca.__path__[0], "training", "managers")],
        class_name=manager_name,
        current_module="yucca.training.managers",
    )

    assert manager, f"searching for {manager_name} " f"but found: {manager}"
    assert issubclass(manager, (YuccaManager, YuccaManager)), "Trainer is not a subclass of YuccaTrainer."

    print(f"{'Using manager: ':25} {manager_name}")
    manager = manager(
        ckpt_path=modelfile,
        enable_logging=False,
        experiment=experiment,
        model_name=model,
        model_dimensions=dimensions,
        task=source_task,
        split_idx=split_idx,
        planner=planner,
        profile=profile,
        batch_size=batch_size,
    )

    # Setting up input paths and output paths
    inpath = join(yucca_raw_data, target_task, "imagesTs")
    ground_truth = join(yucca_raw_data, target_task, "labelsTs")

    outpath = join(
        yucca_results,
        target_task,
        source_task,
        model + "__" + dimensions,
        manager_name + "__" + planner,
        f"fold_{split_idx}",
        f"version_{version}",
        checkpoint,
    )

    if predict_train:
        inpath = join(yucca_raw_data, target_task, "imagesTr")
        ground_truth = join(yucca_raw_data, target_task, "labelsTr")
        outpath += "Tr"

    maybe_mkdir_p(outpath)

    manager.predict_folder(
        inpath,
        disable_tta,
        outpath,
        save_softmax=save_softmax,
        # overwrite=overwrite, # Commented out until overwrite arg is added in manager.
    )

    if isdir(ground_truth) and not no_eval:
        evaluator = YuccaEvaluator(
            manager.model_module.num_classes,
            folder_with_predictions=outpath,
            folder_with_ground_truth=ground_truth,
            task_type=task_type,
            use_wandb=use_wandb,
        )
        evaluator.run()


if __name__ == "__main__":
    main()
