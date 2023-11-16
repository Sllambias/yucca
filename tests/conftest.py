import pytest
from dotenv import load_dotenv, find_dotenv
import os
from pathlib import Path

# @pytest.fixture(scope="session", autouse=True)
# def load_env():
#    env_file = find_dotenv(".env")
#    load_dotenv(env_file)


@pytest.fixture(scope="session", autouse=True)
def set_env():
    os.environ["YUCCA_PREPROCESSED"] = str(Path(__file__).parent.absolute() / "data")
    os.environ["YUCCA_RAW_DATA"] = str(Path(__file__).parent.absolute() / "data")
    os.environ["YUCCA_MODELS"] = str(Path(__file__).parent.absolute() / "data")
    os.environ["YUCCA_RESULTS"] = str(Path(__file__).parent.absolute() / "data")
