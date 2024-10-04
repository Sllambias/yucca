import pytest
import os
import shutil
import subprocess
import torch
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def set_env():
    # Setting the following environment variable means that test data _will not be deleted between tests_.
    # This might screw up some tests down the line, so be careful.
    # PLEASE ONLY SET TO TRUE ON YOUR LOCAL MACHINE FOR DEVELOPMENT PURPOSES AND IF YOU KNOW WHAT YOU ARE DOING
    keep_test_data = os.getenv("YUCCA_KEEP_TEST_DATA", default=False) == "True"

    os.environ["YUCCA_SOURCE"] = str(Path(__file__).parent.absolute() / "data" / "raw_unstructured_data")

    yucca_raw_data = str(Path(__file__).parent.absolute() / "data" / "yucca_raw_data")
    os.environ["YUCCA_RAW_DATA"] = yucca_raw_data
    if os.path.isdir(yucca_raw_data) and not keep_test_data:
        shutil.rmtree(yucca_raw_data)

    yucca_preprocessed_data = str(Path(__file__).parent.absolute() / "data" / "yucca_preprocessed")
    os.environ["YUCCA_PREPROCESSED_DATA"] = yucca_preprocessed_data
    if os.path.isdir(yucca_preprocessed_data) and not keep_test_data:
        shutil.rmtree(yucca_preprocessed_data)

    yucca_models = str(Path(__file__).parent.absolute() / "data" / "yucca_models")
    os.environ["YUCCA_MODELS"] = yucca_models
    if os.path.isdir(yucca_models) and not keep_test_data:
        shutil.rmtree(yucca_models)

    yucca_results = str(Path(__file__).parent.absolute() / "data" / "yucca_results")
    os.environ["YUCCA_RESULTS"] = yucca_results
    if os.path.isdir(yucca_results) and not keep_test_data:
        shutil.rmtree(yucca_results)

    if torch.cuda.is_available():
        accel = "gpu"
    else:
        accel = "cpu"

    os.environ["accelerator"] = accel


@pytest.fixture
def setup_preprocessed_segmentation_data(request):
    from yucca.paths import get_raw_data_path, get_preprocessed_data_path

    subprocess.run(["yucca_convert_task", "-t", "Task000_TEST_SEGMENTATION"], check=True)
    subprocess.run(["yucca_preprocess", "-t", "Task000_TEST_SEGMENTATION"], check=True)

    def finalizer():
        if "YUCCA_KEEP_TEST_DATA" not in os.environ.keys():
            shutil.rmtree(get_raw_data_path())
            shutil.rmtree(get_preprocessed_data_path())

    request.addfinalizer(finalizer)

    return
