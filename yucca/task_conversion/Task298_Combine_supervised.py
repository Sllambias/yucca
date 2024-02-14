import os
from yucca.task_conversion.utils import combine_images_from_tasks, generate_dataset_json
from yucca.paths import yucca_raw_data


def convert(_path: str, _subdir: str = None):
    # Define the name of the new task
    task_name = "Task213_OASIS_Hammers"

    # Define the tasks to combine, such as ["Task001_OASIS", "Task002_LPBA40"]
    # The individual task_conversion scripts must be run prior to executing this, as the script will look for the data in the yucca_raw_data folder.
    tasks_to_combine = ["Task001_OASIS", "Task003_Hammers"]

    target_base = os.path.join(yucca_raw_data, task_name)
    os.makedirs(target_base, exist_ok=True)

    combine_images_from_tasks(tasks=tasks_to_combine, target_base=target_base, run_type="supervised")

    generate_dataset_json(
        os.path.join(target_base, "dataset.json"),
        imagesTr_dir=os.path.join(target_base, "imagesTr"),
        imagesTs_dir=os.path.join(target_base, "imagesTs"),
        modalities=["MRI"],
        labels={0: "background", 1: "Left Hippocampus", 2: "Right Hippocampus"},
        dataset_name=task_name,
        license="CC-BY-SA 4.0",
        dataset_description="Combination of multiple datasets",
        dataset_reference="",
    )
