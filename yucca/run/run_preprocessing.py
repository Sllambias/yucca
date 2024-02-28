import argparse
import yucca
from yucca.utils.task_ids import maybe_get_task_from_task_id
from yucca.utils.files_and_folders import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--task", help="Name of the task to preprocess. " "Should be of format: TaskXXX_MYTASK", required=True
    )
    parser.add_argument(
        "-pl", help="Experiment Planner Class to employ. " "Defaults to the YuccaPlanner", default="YuccaPlanner"
    )
    parser.add_argument(
        "-pr",
        help="Preprocessor Class to employ. "
        "Defaults to the YuccaPreprocessor, but can be ClassificationPreprocessor for classification tasks",
        default="YuccaPreprocessor",
    )
    parser.add_argument(
        "-v",
        help="Designate target view or orientation to obtain with transposition. "
        "Standard settings will handle this for you, but use this to manually specify. "
        "Can be 'X', 'Y' or 'Z'",
    )
    parser.add_argument(
        "--ensemble",
        help="Used to initialize data preprocessing for ensemble/2.5D training",
        default=False,
        action="store_true",
    )
    parser.add_argument("--disable_sanity_checks", help="Enable or disable sanity checks", default=False)
    parser.add_argument("--threads", help="Used to specify the number of processes to use for preprocessing", default=2)
    args = parser.parse_args()

    task = maybe_get_task_from_task_id(args.task)
    planner_name = args.pl
    preprocessor_name = args.pr
    view = args.v
    disable_sanity_checks = args.disable_sanity_checks
    ensemble = args.ensemble
    threads = args.threads

    if not ensemble:
        planner = recursive_find_python_class(
            folder=[join(yucca.__path__[0], "planning")], class_name=planner_name, current_module="yucca.planning"
        )
        planner = planner(task, preprocessor_name, threads=threads, disable_sanity_checks=disable_sanity_checks, view=view)
        planner.plan()
        planner.preprocess()
    if ensemble:
        views = ["X", "Y", "Z"]
        for view in views:
            planner = recursive_find_python_class(
                folder=[join(yucca.__path__[0], "planning")], class_name=planner_name, current_module="yucca.planning"
            )
            planner = planner(task, preprocessor_name, threads=threads, disable_sanity_checks=disable_sanity_checks, view=view)
            planner.plan()
            planner.preprocess()


if __name__ == "__main__":
    main()
