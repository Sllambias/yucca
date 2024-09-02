"""
PLEASE READ YUCCA/DOCUMENTATION/TUTORIALS/ENVIRONMENT_VARIABLES.MD FOR INFORMATION ON HOW TO SET THIS UP
"""

import os
from dotenv import load_dotenv

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p as ensure_dir_exists


def var_is_set(var):
    return var in os.environ.keys()


def get_environment_variable(var):
    load_dotenv()
    if not var_is_set(var):
        raise ValueError("Missing required environment variable {YUCCA_SOURCE}.")

    path = os.environ[var]
    ensure_dir_exists(path)
    return path


def get_yucca_source():
    return get_environment_variable("YUCCA_SOURCE")


def get_yucca_raw_data():
    return get_environment_variable("YUCCA_RAW_DATA")


def get_yucca_preprocessed_data():
    return get_environment_variable("YUCCA_PREPROCESSED_DATA")


def get_yucca_models():
    return get_environment_variable("YUCCA_MODELS")


def get_yucca_results():
    return get_environment_variable("YUCCA_RESULTS")


def get_yucca_wandb_entity():
    return os.getenv("YUCCA_WANDB_ENTITY")
