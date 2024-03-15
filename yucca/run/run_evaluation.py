"""
This script can be used in two ways:
1. You can specify the paths to the folder containing predictions and labels/ground truth
using "--pred" and "--gt" and define the labels of interest using e.g. "-c 0 1"

2. You can specify the task, trainer and planner like it's also done in other yucca_ scripts.
if "-t" for target task is left blank, we assume you are predicting and evaluating data
from the same task as the one the model is trained on.
"""

import argparse
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from yucca.evaluation.YuccaEvaluator import YuccaEvaluator
from yucca.utils.task_ids import maybe_get_task_from_task_id
from yucca.paths import yucca_raw_data, yucca_results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        help="Name of the source task i.e. what the model is trained on. " "Should be of format: TaskXXX_MYTASK",
        required=False,
    )
    parser.add_argument(
        "-t",
        help="Name of the target task i.e. the data to be predicted. " "Should be of format: TaskXXX_MYTASK",
        required=False,
    )
    parser.add_argument("-m", help="Model Architecture. Defaults to UNet.", default="UNet")
    parser.add_argument("-d", help="2D, 25D or 3D model. Defaults to 3D.", default="3D")
    parser.add_argument(
        "-man",
        help="Full name of Manager Class. \n" "e.g. 'YuccaTrainer_DCE' or 'YuccaTrainer'. Defaults to YuccaTrainer.",
        default="YuccaManager",
    )
    parser.add_argument("-pl", help="Plan ID. Defaults to YuccaPlanner", default="YuccaPlanner")
    parser.add_argument("-chk", help="Checkpoint used for inference. Defaults to best.", default="best")
    parser.add_argument("-c", nargs="*", help="Classes to include for evaluation", type=str)

    # Alternatively, these can be used to manually specify paths
    parser.add_argument("--pred", help="manually specify path to predicted segmentations", default=None, required=False)
    parser.add_argument("--gt", help="manually specify path to ground truth", default=None, required=False)

    # Optionals (infrequently changed)
    parser.add_argument(
        "--task_type",
        default="segmentation",
        type=str,
        required=False,
        help="Defaults to segmentation. Set to 'classification' for classification tasks.",
    )
    parser.add_argument(
        "--as_binary", help="run evaluation as if data was binary", action="store_true", default=None, required=False
    )
    parser.add_argument(
        "--experiment",
        help="A name for the experiment being performed, with no spaces.",
        default="default",
    )
    parser.add_argument(
        "--no_wandb",
        help="Disable logging of evaluation results to wandb",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument("--obj_eval", help="enable object evaluation", action="store_true", default=None, required=False)
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        required=False,
        help="Overwrite existing predictions",
    )
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
        "--version", "-v", help="version number of the model. Defaults to 0.", default=0, type=int, required=False
    )
    args = parser.parse_args()

    source_task = maybe_get_task_from_task_id(args.s)
    target_task = maybe_get_task_from_task_id(args.t)
    manager_name = args.man
    model = args.m
    dimensions = args.d
    plan_id = args.pl
    checkpoint = args.chk
    obj = args.obj_eval
    overwrite = args.overwrite
    as_binary = args.as_binary
    classes = args.c
    predpath = args.pred
    gtpath = args.gt
    num_version = args.version
    task_type = args.task_type
    use_wandb = not args.no_wandb
    split_idx = args.split_idx
    split_data_method = args.split_data_method
    split_data_param = args.split_data_param
    surface_eval = args.surface_eval
    assert (predpath and gtpath) or source_task, "Either supply BOTH paths or the source task"

    if not predpath:
        if not target_task:
            target_task = source_task

        predpath = join(  # TODO: Extract this into a function
            yucca_results,
            target_task,
            source_task,
            model + "__" + dimensions,
            manager_name + "__" + plan_id,
            f"{split_data_method}_{split_data_param}_fold_{split_idx}",
            "version_" + str(num_version),
            checkpoint,
        )
        gtpath = join(yucca_raw_data, target_task, "labelsTs")
        classes = list(load_json(join(yucca_raw_data, target_task, "dataset.json"))["labels"].keys())

    evaluator = YuccaEvaluator(
        classes,
        folder_with_predictions=predpath,
        folder_with_ground_truth=gtpath,
        do_object_eval=obj,
        do_surface_eval=surface_eval,
        overwrite=overwrite,
        as_binary=as_binary,
        task_type=task_type,
        use_wandb=use_wandb,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
