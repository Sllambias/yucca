import pytest
import os
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def set_env():
    os.environ["YUCCA_PREPROCESSED_DATA"] = str(Path(__file__).parent.absolute() / "data")
    os.environ["YUCCA_RAW_DATA"] = str(Path(__file__).parent.absolute() / "data")
    os.environ["YUCCA_MODELS"] = str(Path(__file__).parent.absolute() / "data")
    os.environ["YUCCA_RESULTS"] = str(Path(__file__).parent.absolute() / "data")
