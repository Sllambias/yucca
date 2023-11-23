import os
from yucca.task_conversion.utils import combine_imagesTr_from_tasks, generate_dataset_json
from yucca.paths import yucca_raw_data


# Define the name of the new task
task_name = "TaskXXX_MyCombinedTask"

# Define the expected labels (leave empty for no labels)
expected_labels = {}

# Define expected modalities, such as ('MRI', 'CT',) or just ('Medical Images',) - remember trailing "," for singular modalities
expected_modalities = ("Unspecified",)

# Define the tasks to combine, such as ["Task001_OASIS", "Task002_LPBA40"]
# The individual task_conversion scripts must be run prior to executing this, as the script will look for the data in the yucca_raw_data folder.
tasks_to_combine = []

### In most cases the remaining can be left untouched ###
# Setting the paths to save the new task and making the directories
target_base = os.path.join(yucca_raw_data, task_name)
target_imagesTr = os.path.join(yucca_raw_data, task_name, "imagesTr")
target_imagesTs = None
os.makedirs(target_imagesTr, exist_ok=True)

combine_imagesTr_from_tasks(tasks=tasks_to_combine, task_name=task_name, target_dir=target_imagesTr)

generate_dataset_json(
    os.path.join(target_base, "dataset.json"),
    target_imagesTr,
    target_imagesTs,
    modalities=expected_modalities,
    labels=expected_labels,
    dataset_name=task_name,
    license="CC-BY-SA 4.0",
    dataset_description="",
    dataset_reference="",
)
