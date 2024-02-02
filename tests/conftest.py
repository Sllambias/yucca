import pytest
import os
import shutil
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def set_env():
    os.environ["YUCCA_SOURCE"] = str(Path(__file__).parent.absolute() / "data" / "raw_unstructured_data")

    yucca_raw_data = str(Path(__file__).parent.absolute() / "data" / "yucca_raw_data")
    os.environ["YUCCA_RAW_DATA"] = str(Path(__file__).parent.absolute() / "data" / "yucca_raw_data")
    if os.path.isdir(yucca_raw_data):
        shutil.rmtree(yucca_raw_data)

    yucca_preprocessed_data = str(Path(__file__).parent.absolute() / "data" / "yucca_preprocessed")
    os.environ["YUCCA_PREPROCESSED_DATA"] = yucca_preprocessed_data
    if os.path.isdir(yucca_preprocessed_data):
        shutil.rmtree(yucca_preprocessed_data)

    yucca_models = str(Path(__file__).parent.absolute() / "data" / "yucca_models")
    os.environ["YUCCA_MODELS"] = yucca_models
    if os.path.isdir(yucca_models):
        shutil.rmtree(yucca_models)

    yucca_results = str(Path(__file__).parent.absolute() / "data" / "yucca_results")
    os.environ["YUCCA_RESULTS"] = yucca_results
    if os.path.isdir(yucca_results):
        shutil.rmtree(yucca_results)
