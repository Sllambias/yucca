from dataclasses import dataclass


@dataclass
class TaskConfig:
    continue_from_most_recent: bool
    manager_name: str
    model_dimensions: str
    model_name: str
    planner_name: str
    split_idx: int
    task: str
    experiment: str

    def lm_hparams(self):
        return {
            "continue_from_most_recent": self.continue_from_most_recent,
            "manager_name": self.manager_name,
            "model_dimensions": self.model_dimensions,
            "model_name": self.model_name,
            "planner_name": self.planner_name,
            "task": self.task,
            "split_idx": self.split_idx,
            "experiment": self.experiment,
        }


def get_task_config(
    task: str,
    continue_from_most_recent: bool = True,
    manager_name: str = "YuccaManager",
    model_dimensions: str = "3D",
    model_name: str = "UNet",
    planner_name: str = "YuccaPlanner",
    split_idx: int = 0,
    experiment: str = "default",
):
    assert model_dimensions is not None

    return TaskConfig(
        continue_from_most_recent=continue_from_most_recent,
        manager_name=manager_name,
        model_dimensions=model_dimensions,
        model_name=model_name,
        planner_name=planner_name,
        split_idx=split_idx,
        task=task,
        experiment=experiment,
    )
