import argparse
import yucca
from yucca.pipeline.task_conversion.utils import get_task_from_task_id
from yucca.paths import (
    get_raw_data_path,
    get_results_path,
    get_models_path,
    get_preprocessed_data_path,
)
from yucca.pipeline.evaluation.YuccaEvaluator import YuccaEvaluator
from yucca.pipeline.managers.YuccaManager import YuccaManager
from yucca.functional.utils.files_and_folders import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    isfile,
    maybe_mkdir_p as ensure_dir_exists,
    isdir,
    subdirs,
    load_pickle,
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
    parser.add_argument("-m", "--model", help="Model Architecture. Defaults to UNet.", default="UNet")
    parser.add_argument(
        "-man",
        "--manager",
        help="Full name of Trainer Class. \n" "e.g. 'YuccaTrainer_DCE' or 'YuccaTrainer'. Defaults to YuccaTrainer.",
        default="YuccaManager",
    )
    parser.add_argument("-pl", "--planner", help="Planner. Defaults to YuccaPlanner", default="YuccaPlanner")
    parser.add_argument("--split_idx", type=int, help="idx of splits to use for training.", default=0)
    parser.add_argument(
        "--split_data_method", help="Specify splitting method. Either kfold, simple_train_val_split", default="kfold"
    )
    parser.add_argument(
        "--split_data_param",
        help="Specify the parameter for the selected split method. For KFold use an int, for simple_split use a float between 0.0-1.0.",
        default=5,
    )
    parser.add_argument("--surface_eval", help="enable surface evaluation", action="store_true", default=None, required=False)
    parser.add_argument(
        "--surface_tol", type=int, help="controls the tolerance (in mm) of the surface_evaluation", default=2, required=False
    )
    parser.add_argument(
        "--task_type",
        default="segmentation",
        type=str,
        required=False,
        help="Defaults to segmentation. Set to 'classification' for classification tasks.",
    )
    parser.add_argument(
        "-v",
        "--version",
        help="Version to use for inference. Defaults to the newest version.",
        default=None,
    )

    # Alternatively, these can be used to manually specify paths
    parser.add_argument("--pred", help="manually specify path to predicted segmentations", default=None, required=False)
    parser.add_argument("--gt", help="manually specify path to ground truth", default=None, required=False)

    # Optionals (occasionally changed)
    parser.add_argument(
        "--experiment",
        help="A name for the experiment being performed, with no spaces.",
        default="default",
    )

    parser.add_argument(
        "--disable_preprocessing",
        help="Used to disable preprocessing for already preprocessed test data. Use with caution.",
        default=False,
        action="store_true",
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
        "--overwrite",
        default=False,
        action="store_true",
        required=False,
        help="Overwrite existing predictions",
    )
    parser.add_argument(
        "--predict_train",
        default=False,
        action="store_true",
        required=False,
        help="Predict on the training set. Useful for debugging.",
    )
    parser.add_argument(
        "--predict_val",
        default=False,
        action="store_true",
        required=False,
        help="Predict on the validation set. Uses the split method, param and idx already supplied.",
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
    parser.add_argument("--ext", help="file extension e.g. '.nii.gz', '.png' or '.txt'")

    args = parser.parse_args()

    # Required
    source_task = get_task_from_task_id(args.s, stage="models")
    target_task = get_task_from_task_id(args.t, stage="raw")

    # Optionals (frequently changed)
    checkpoint = args.checkpoint
    dimensions = args.dimensions
    manager_name = args.manager
    model = args.model
    planner = args.planner
    split_idx = args.split_idx
    split_data_method = args.split_data_method
    split_data_param = args.split_data_param
    surface_eval = args.surface_eval
    surface_tol = args.surface_tol
    task_type = args.task_type
    version = args.version

    # Alternative manual paths
    predpath = args.pred
    gtpath = args.gt

    # Optionals (occasionally changed)
    experiment = args.experiment
    disable_tta = args.disable_tta
    disable_preprocessing = args.disable_preprocessing
    no_eval = args.no_eval
    overwrite = args.overwrite
    predict_train = args.predict_train
    predict_val = args.predict_val
    profile = args.profile
    save_softmax = args.save_softmax
    use_wandb = not args.no_wandb
    ext = args.ext

    kwargs = {}
    strict = True
    split = None

    path_to_versions = join(
        get_models_path(),
        source_task,
        model + "__" + dimensions,
        manager_name + "__" + planner,
        experiment,
        f"{split_data_method}_{split_data_param}_fold_{split_idx}",
    )
    if version is None:
        versions = [int(i.split("_")[-1]) for i in subdirs(path_to_versions, join=False)]
        version = str(max(versions))

    if predict_val:
        checkpoint = "last"

    modelfile = join(
        get_models_path(),
        source_task,
        model + "__" + dimensions,
        manager_name + "__" + planner,
        experiment,
        f"{split_data_method}_{split_data_param}_fold_{split_idx}",
        f"version_{version}",
        "checkpoints",
        checkpoint + ".ckpt",
    )

    assert isfile(modelfile), "Can't find .cpkt file with trained model weights. " f"Should be located at: {modelfile}"
    print(f"######################################################################## \n" f"{'Using model: ':25} {modelfile}")

    manager = recursive_find_python_class(
        folder=[join(yucca.__path__[0], "pipeline", "managers")],
        class_name=manager_name,
        current_module="yucca.pipeline.managers",
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
        split_data_method=split_data_method,
        split_data_param=split_data_param,
        planner=planner,
        profile=profile,
        **kwargs,
    )

    # Setting up input paths and output paths
    inpath = join(get_raw_data_path(), target_task, "imagesTs") if not predpath else predpath
    ground_truth = join(get_raw_data_path(), target_task, "labelsTs") if not gtpath else gtpath

    outpath = join(
        get_results_path(),
        target_task,
        source_task,
        model + "__" + dimensions,
        manager_name + "__" + planner,
        experiment,
        f"{split_data_method}_{split_data_param}_fold_{split_idx}",
        f"version_{version}",
        checkpoint,
    )

    if predict_train:
        inpath = join(get_raw_data_path(), target_task, "imagesTr")
        ground_truth = join(get_raw_data_path(), target_task, "labelsTr")
        outpath += "Tr"
    elif predict_val:
        inpath = join(get_raw_data_path(), target_task, "imagesTr")
        ground_truth = join(get_raw_data_path(), target_task, "labelsTr")
        outpath += "Val"
        split = load_pickle(join(get_preprocessed_data_path(), source_task, "splits.pkl"))
        split = split[str(split_data_method)][split_data_param][split_idx]["val"]
        strict = False

    ensure_dir_exists(outpath)

    if disable_preprocessing:
        predict_fn = manager.predict_preprocessed_folder
    else:
        predict_fn = manager.predict_folder

    predict_fn(
        inpath,
        disable_tta,
        overwrite_predictions=overwrite,
        output_folder=outpath,
        pred_include_cases=split,
        save_softmax=save_softmax,
    )

    if isdir(ground_truth) and not no_eval:
        evaluator = YuccaEvaluator(
            manager.plan_config.labels,
            folder_with_predictions=outpath,
            folder_with_ground_truth=ground_truth,
            do_surface_eval=surface_eval,
            overwrite=overwrite,
            regions=manager.plan_config.regions if manager.plan_config.use_label_regions else None,
            surface_tol=surface_tol,
            task_type=task_type,
            use_wandb=use_wandb,
            strict=strict,
            extension=ext,
        )
        evaluator.run()


if __name__ == "__main__":
    main()
