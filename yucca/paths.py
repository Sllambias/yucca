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
        raise ValueError(f"Missing required environment variable {var}.")

    path = os.environ[var]
    ensure_dir_exists(path)
    return path


def get_source_path():
    return get_environment_variable("YUCCA_SOURCE")


def get_raw_data_path():
    return get_environment_variable("YUCCA_RAW_DATA")


def get_preprocessed_data_path():
    return get_environment_variable("YUCCA_PREPROCESSED_DATA")


def get_models_path():
    return get_environment_variable("YUCCA_MODELS")


def get_results_path():
    return get_environment_variable("YUCCA_RESULTS")


def get_yucca_wandb_entity():
    return os.getenv("YUCCA_WANDB_ENTITY")
