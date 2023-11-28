import argparse
import yucca
from yucca.utils.task_ids import maybe_get_task_from_task_id
from yucca.paths import yucca_raw_data, yucca_results, yucca_models
from yucca.evaluation.YuccaEvaluator import YuccaEvaluator
from yucca.training.trainers.YuccaManager import YuccaManager
from yucca.training.trainers.YuccaLightningManager import YuccaLightningManager
from yuccalib.utils.files_and_folders import (
    recursive_find_python_class,
    merge_softmax_from_folders,
)
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
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
    parser.add_argument(
        "-f",
        help="Select the fold that was used to train the model desired for inference. "
        "Defaults to looking for a model trained on fold 0.",
        default="0",
    )
    parser.add_argument("-m", help="Model Architecture. Defaults to UNet.", default="UNet")
    parser.add_argument("-d", help="2D or 3D model. Defaults to 3D.", default="3D")
    parser.add_argument(
        "-tr",
        help="Full name of Trainer Class. \n" "e.g. 'YuccaTrainer_DCE' or 'YuccaTrainer'. Defaults to YuccaTrainer.",
        default="YuccaTrainer",
    )
    parser.add_argument("-pl", help="Planner. Defaults to YuccaPlanner", default="YuccaPlanner")
    parser.add_argument(
        "-chk",
        help="Checkpoint to use for inference. Defaults to model_best.",
        default="best",
        default="best",
    )
    parser.add_argument(
        "-v",
        help="Version to use for inference. Defaults to the newest version.",
        default=None,
    )
    parser.add_argument(
        "--ensemble",
        help="Used to initialize data preprocessing for ensemble/2.5D training",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--disable_tta",
        "--profile",
        help="Used to enable inference profiling",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--not_strict",
        default=False,
        action="store_true",
        required=False,
        help="Strict determines if all expected modalities must be present, "
        "with the appropriate suffixes (e.g. '_000.nii.gz'). "
        "Only touch if you know what you're doing.",
    )
    parser.add_argument(
        "--save_softmax",
        default=False,
        action="store_true",
        required=False,
        help="Save softmax outputs. Required for softmax fusion.",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        required=False,
        help="Overwrite existing predictions",
    )
    parser.add_argument(
        "--no_eval",
        help="Disable evaluation and creation of metrics file (result.json)",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--predict_train",
        default=False,
        action="store_true",
        required=False,
        help="Predict on the training set. Useful for debugging.",
    )
    # parser.add_argument("--threads", help="number of threads/processes", default=2)

    args = parser.parse_args()

    source_task = maybe_get_task_from_task_id(args.s)
    target_task = maybe_get_task_from_task_id(args.t)
    manager_name = args.tr
    model = args.m
    dimensions = args.d
    folds = args.f
    planner = args.pl
    profile = args.profile
    checkpoint = args.chk
    version = args.v
    ensemble = args.ensemble
    disable_tta = args.disable_tta
    not_strict = args.not_strict
    save_softmax = args.save_softmax
    overwrite = args.overwrite
    no_eval = args.no_eval
    predict_train = args.predict_train
    # threads = args.threads

    folders_with_softmax = []
    if ensemble:
        print("Running ensemble inference on the default ensemble plans \n" "Save_softmax set to True.")
        plans = [planner + "X", planner + "Y", planner + "Z"]
        save_softmax = True
    else:
        plans = [planner]

    for planner in plans:
        path_to_versions = join(
            yucca_models, source_task, model + "__" + dimensions, manager_name + "__" + planner, f"fold_{folds}"
        )
        if version is None:
            versions = [int(i.split("_")[-1]) for i in subdirs(path_to_versions, join=False)]
            version = str(max(versions))
        modelfile = join(
            yucca_models,
            source_task,
            model + "__" + dimensions,
            manager_name + "__" + planner,
            f"fold_{folds}",
            f"version_{version}",
            "checkpoints",
            checkpoint + ".ckpt",
        )

        assert isfile(modelfile), "Can't find .cpkt file with trained model weights. " f"Should be located at: {modelfile}"
        print(
            f"######################################################################## \n" f"{'Using model: ':25} {modelfile}"
        )

        manager = recursive_find_python_class(
            folder=[join(yucca.__path__[0], "training")],
            class_name=manager_name,
            current_module="yucca.training",
        )

        assert manager, f"searching for {manager_name} " f"but found: {manager}"
        assert issubclass(manager, (YuccaManager, YuccaLightningManager)), "Trainer is not a subclass of YuccaTrainer."

        print(f"{'Using manager: ':25} {manager_name}")
        manager = manager(
            disable_logging=False,
            model_name=model,
            model_dimensions=dimensions,
            task=source_task,
            folds=folds,
            planner=planner,
            ckpt_path=modelfile,
            profile=profile,
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
            f"fold_{folds}",
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
            # overwrite=overwrite, # Commented out until overwrite arg is added in manager.
        )

        folders_with_softmax.append(outpath)

        if isdir(ground_truth) and not no_eval:
            evaluator = YuccaEvaluator(
                manager.model_module.num_classes,
                folder_with_predictions=outpath,
                folder_with_ground_truth=ground_truth,
            )
            evaluator.run()

    if ensemble:
        ensemble_outpath = join(
            yucca_results,
            target_task,
            source_task,
            model + dimensions,
            manager_name + "__" + planner + "_Ensemble",
            "fold_" + folds + "_" + checkpoint,
        )
        merge_softmax_from_folders(folders_with_softmax, ensemble_outpath)

        if isdir(ground_truth) and not no_eval:
            evaluator = YuccaEvaluator(
                manager.classes,
                folder_with_predictions=ensemble_outpath,
                folder_with_ground_truth=ground_truth,
            )
            evaluator.run()


if __name__ == "__main__":
    main()
