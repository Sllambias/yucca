"""
PLEASE READ YUCCA/DOCUMENTATION/TUTORIALS/ENVIRONMENT_VARIABLES.MD FOR INFORMATION ON HOW TO SET THIS UP
"""

import os
import warnings
from dotenv import load_dotenv

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

load_dotenv()

vars = ["YUCCA_RAW_DATA", "YUCCA_PREPROCESSED_DATA", "YUCCA_MODELS", "YUCCA_RESULTS"]
vals = {}

for var in vars:
    if var in os.environ.keys():
        vals[var] = os.environ[var]
        maybe_mkdir_p(vals[var])
    else:
        warnings.warn(f"Missing environment variable {var}.")
        vals[var] = None

yucca_raw_data = vals["YUCCA_RAW_DATA"]
yucca_preprocessed_data = vals["YUCCA_PREPROCESSED_DATA"]
yucca_models = vals["YUCCA_MODELS"]
yucca_results = vals["YUCCA_RESULTS"]

if "YUCCA_SOURCE" in os.environ.keys():
    yucca_source = os.environ["YUCCA_SOURCE"]
else:
    yucca_source = None

if "YUCCA_WANDB_ENTITY" in os.environ.keys():
    yucca_wandb_entity = os.environ["YUCCA_WANDB_ENTITY"]
else:
    yucca_wandb_entity = None
