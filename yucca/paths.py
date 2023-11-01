"""
PLEASE READ YUCCA/DOCUMENTATION/TUTORIALS/ENVIRONMENT_VARIABLES.MD FOR INFORMATION ON HOW TO SET THIS UP
"""
import os
from dotenv import load_dotenv

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

assert load_dotenv()

yucca_raw_data = os.environ['YUCCA_RAW_DATA'] if "YUCCA_RAW_DATA" in os.environ.keys() else None
yucca_preprocessed = os.environ['YUCCA_PREPROCESSED'] if "YUCCA_PREPROCESSED" in os.environ.keys() else None
yucca_models = os.environ['YUCCA_MODELS'] if "YUCCA_MODELS" in os.environ.keys() else None
yucca_results = os.environ['YUCCA_RESULTS'] if "YUCCA_RESULTS" in os.environ.keys() else None

if yucca_raw_data is not None:
    maybe_mkdir_p(yucca_raw_data)
else:
    print("yucca_raw_data is not defined and Yucca might not work")
    yucca_raw_data = None

if yucca_preprocessed is not None:
    maybe_mkdir_p(yucca_preprocessed)
else:
    print("yucca_preprocessing_dir is not defined and Yucca might not work")
    yucca_preprocessed = None

if yucca_models is not None:
    maybe_mkdir_p(yucca_models)
else:
    print("yucca_models is not defined and Yucca might not work")
    yucca_models = None

if yucca_results is not None:
    maybe_mkdir_p(yucca_results)
else:
    print("yucca_results is not defined and Yucca might not work")
    yucca_results = None
