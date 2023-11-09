import argparse
import yucca
from yucca.utils.task_ids import maybe_get_task_from_task_id
from yuccalib.utils.files_and_folders import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--task", help="Name of the task to preprocess. " "Should be of format: TaskXXX_MYTASK", required=True
    )
    parser.add_argument(
        "-pl", help="Experiment Planner Class to employ. " "Defaults to the YuccaPlannerV2", default="YuccaPlanner"
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
    parser.add_argument("--disable_unit_tests", help="Enable or disable unittesting", default=False)
    parser.add_argument("--multi_task", help="Enable multi task preprocessing", default=False, action="store_true")

    # parser.add_argument("--threads", help="Number of threads/processes. \n"
    #                    "Don't touch this unless you know what you're doing.", default=2)

    args = parser.parse_args()

    task = maybe_get_task_from_task_id(args.task)
    planner_name = args.pl
    view = args.v
    disable_testing = args.disable_unit_tests
    ensemble = args.ensemble
    multi_task = args.multi_task
    # threads = args.threads

    if not ensemble:
        planner = recursive_find_python_class(
            folder=[join(yucca.__path__[0], "planning")], class_name=planner_name, current_module="yucca.planning"
        )
        planner = planner(task, 2, disable_testing, view=view, multi_task=multi_task)
        planner.plan()
        planner.preprocess()
    if ensemble:
        views = ["X", "Y", "Z"]
        for view in views:
            planner = recursive_find_python_class(
                folder=[join(yucca.__path__[0], "planning")], class_name=planner_name, current_module="yucca.planning"
            )
            planner = planner(task, 2, disable_testing, view=view)
            planner.plan()
            planner.preprocess()


if __name__ == "__main__":
    main()
