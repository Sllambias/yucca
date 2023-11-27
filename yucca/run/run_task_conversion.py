import argparse
from yucca.paths import yucca_source
import importlib
import re


def remove_task_prefix(str):
    """
    Removes the "TaskXXX_" prefix from string. Number is arbitrary, and T need not be capitalized.
    Example:
      Input: "Task001_OASIS_IS_COOL"
      Output: "OASIS_IS_COOL"
    """
    pattern = re.compile(r"^Task\d+_", re.IGNORECASE)
    return pattern.sub("", str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--task", help="Name of the task to preprocess. " "Should be of format: TaskXXX_MYTASK", required=True
    )
    parser.add_argument("-p", "--path", help="Path to source data", default=yucca_source)
    parser.add_argument("-d", "--subdir", help="Directory of data inside source data")

    args = parser.parse_args()
    task_converter = importlib.import_module(f"yucca.task_conversion.{args.task}")

    if args.subdir is None:
        task_converter.convert(args.path)
    else:
        task_converter.convert(args.path, args.subdir)


if __name__ == "__main__":
    main()
