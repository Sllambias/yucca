import os
from yucca.task_conversion.utils import combine_imagesTr_from_tasks, generate_dataset_json
from yucca.paths import yucca_raw_data


def convert(path: str, subdir: str = None):
    # Define the name of the new task
    task_name = "Task299_Combine"

    # Define the expected labels (leave empty for no labels)
    expected_labels = {}

    # Define the tasks to combine, such as ["Task001_OASIS", "Task002_LPBA40"]
    # The individual task_conversion scripts must be run prior to executing this, as the script will look for the data in the yucca_raw_data folder.
    tasks_to_combine = ["Task200_PPMI", "Task203_OASIS3", "Task205_Hippocampus", "Task206_BrainTumour"]

    ### In most cases the remaining can be left untouched ###
    # Setting the paths to save the new task and making the directories
    target_base = os.path.join(yucca_raw_data, task_name)
    target_imagesTr = os.path.join(yucca_raw_data, task_name, "imagesTr")

    os.makedirs(target_imagesTr, exist_ok=True)

    combine_imagesTr_from_tasks(tasks=tasks_to_combine, target_dir=target_imagesTr)

    generate_dataset_json(
        os.path.join(target_base, "dataset.json"),
        target_imagesTr,
        imagesTs_dir=None,
        modalities=["t1", "t2", "mprage", "flair", "gre", "dwi", "swi", "grappa"],
        labels=expected_labels,
        dataset_name=task_name,
        license="CC-BY-SA 4.0",
        dataset_description="Combination of multiple datasets",
        dataset_reference="",
    )
