"""
PLEASE READ YUCCA/DOCUMENTATION/TUTORIALS/ENVIRONMENT_VARIABLES.MD FOR INFORMATION ON HOW TO SET THIS UP
"""
import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

yucca_raw_data = os.environ['yucca_raw_data'] if "yucca_raw_data" in os.environ.keys() else None
yucca_preprocessed = os.environ['yucca_preprocessed'] if "yucca_preprocessed" in os.environ.keys() else None
yucca_models = os.environ['yucca_models'] if "yucca_models" in os.environ.keys() else None
yucca_results = os.environ['yucca_results'] if "yucca_results" in os.environ.keys() else None

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
