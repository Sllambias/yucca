"""
This script can be used in two ways:
1. You can specify the paths to the folder containing predictions and labels/ground truth
using "--pred" and "--gt" and define the labels of interest using e.g. "-c 0 1"

2. You can specify the task, trainer and planner like it's also done in other yucca_ scripts.
if "-t" for target task is left blank, we assume you are predicting and evaluating data
from the same task as the one the model is trained on.
"""

import argparse
from yucca.evaluation.YuccaEvaluator import YuccaEvaluator
from yucca.utils.task_ids import maybe_get_task_from_task_id
from yucca.paths import yucca_raw_data, yucca_results
from batchgenerators.utilities.file_and_folder_operations import load_json, join


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
    parser.add_argument(
        "-f",
        help="Select the fold that was used to train the model desired for inference. "
        "Defaults to looking for a model trained on fold 0.",
        default="0",
    )
    parser.add_argument("-m", help="Model Architecture. Defaults to UNet.", default="UNet")
    parser.add_argument("-d", help="2D, 25D or 3D model. Defaults to 3D.", default="3D")
    parser.add_argument(
        "-tr",
        help="Full name of Trainer Class. \n" "e.g. 'YuccaTrainer_DCE' or 'YuccaTrainer'. Defaults to YuccaTrainer.",
        default="YuccaTrainer",
    )
    parser.add_argument("-pl", help="Plan ID. Defaults to YuccaPlanner", default="YuccaPlanner")
    parser.add_argument(
        "-chk", help="Checkpoint to use for inference. Defaults to checkpoint_best.", default="checkpoint_best"
    )

    parser.add_argument("-c", nargs="*", help="Classes to include for evaluation", type=str)
    parser.add_argument("--pred", help="path to predicted segmentations", default=None, required=False)
    parser.add_argument("--gt", help="path to ground truth", default=None, required=False)
    parser.add_argument("--obj_eval", help="enable object evaluation", action="store_true", default=None, required=False)
    parser.add_argument(
        "--as_binary", help="run evaluation as if data was binary", action="store_true", default=None, required=False
    )

    args = parser.parse_args()

    source_task = maybe_get_task_from_task_id(args.s)
    target_task = maybe_get_task_from_task_id(args.t)
    trainer_name = args.tr
    model = args.m
    dimensions = args.d
    folds = args.f
    plan_id = args.pl
    checkpoint = args.chk
    obj = args.obj_eval
    as_binary = args.as_binary
    classes = args.c
    predpath = args.pred
    gtpath = args.gt

    assert (predpath and gtpath) or source_task, "Either supply BOTH paths or the source task"

    if not predpath:
        if not target_task:
            target_task = source_task

        predpath = join(
            yucca_results,
            target_task,
            source_task,
            model + dimensions,
            trainer_name + "__" + plan_id,
            "fold_" + folds + "_" + checkpoint,
        )
        gtpath = join(yucca_raw_data, target_task, "labelsTs")
        classes = list(load_json(join(yucca_raw_data, target_task, "dataset.json"))["labels"].keys())

    evaluator = YuccaEvaluator(
        classes, folder_with_predictions=predpath, folder_with_ground_truth=gtpath, do_object_eval=obj, as_binary=as_binary
    )
    evaluator.run()


if __name__ == "__main__":
    main()
