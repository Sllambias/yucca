import os
import subprocess


def test_task_convert():
    subprocess.run(["yucca_convert_task", "-t", "Task000_TEST_SEGMENTATION"], check=True)


def test_functional_preprocessing():
    subprocess.run(["python", "yucca/documentation/templates/functional_preprocessing.py"], check=True)


def test_functional_training():
    subprocess.run(["python", "yucca/documentation/templates/functional_training.py"], check=True)


def test_functional_inference():
    subprocess.run(["python", "yucca/documentation/templates/functional_inference.py"], check=True)
