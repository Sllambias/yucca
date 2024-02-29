"""
This script can be used in two ways:
1. You can specify the paths to the folder containing predictions and labels/ground truth
using "--pred" and "--gt" and define the labels of interest using e.g. "-c 0 1"

2. You can specify the task, trainer and planner like it's also done in other yucca_ scripts.
if "-t" for target task is left blank, we assume you are predicting and evaluating data
from the same task as the one the model is trained on.
"""

import argparse
from yucca.utils.saving import merge_softmax_from_folders


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_dirs",
        help="",
        required=True,
        nargs="*",
    )
    parser.add_argument(
        "--out_dir",
        help="Name of the target task i.e. the data to be predicted. " "Should be of format: TaskXXX_MYTASK",
        required=False,
    )
    args = parser.parse_args()

    merge_softmax_from_folders(folders=args.in_dirs, outpath=args.out_dir)


if __name__ == "__main__":
    main()
