from yucca.paths import yucca_raw_data
from batchgenerators.utilities.file_and_folder_operations import subdirs


def maybe_get_task_from_task_id(task_id: str | int):
    task_id = str(task_id)
    tasks = subdirs(yucca_raw_data, join=False)

    # Check if name is already complete
    if task_id in tasks:
        return task_id

    # If not, we try to recreate the name
    # We use the raw_data folder as reference
    for task in tasks:
        if task_id.lower() in task.lower():
            return task

    # If we can't find anything we just return the original, on the offchance that the task does not exist in Raw Data while existing in e.g. Preprocessed
    print(
        f"Couldn't find a task called: {task_id} in the raw data folder: {yucca_raw_data}. If your task only exists in e.g. the Preprocessed folder things might still work."
    )
    return task_id
