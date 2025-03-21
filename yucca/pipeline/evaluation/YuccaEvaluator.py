import os
from typing import Literal, Optional, Union
import numpy as np
import json
import wandb
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, load_json, isfile
from yucca.functional.evaluation.metrics import (
    dice,
    jaccard,
    sensitivity,
    precision,
    TP,
    FP,
    FN,
    total_pos_gt,
    total_pos_pred,
    volume_similarity,
    accuracy,
)
from yucca.functional.evaluation.evaluate_folder import (
    evaluate_folder_segm,
    evaluate_folder_cls,
)
from yucca.paths import get_raw_data_path


class YuccaEvaluator(object):
    def __init__(
        self,
        labels: Union[dict, int],
        folder_with_predictions,
        folder_with_ground_truth,
        use_wandb: bool,
        as_binary=False,
        do_object_eval=False,
        do_surface_eval=False,
        regions: Optional[dict] = None,
        overwrite: bool = False,
        surface_tol: int = 1,
        task_type: Literal["segmentation", "classification", "regression"] = "segmentation",
        extension: str = None,
        strict: bool = True,
    ):
        self.name = "results"

        self.labels = labels
        self.regions = regions
        self.overwrite = overwrite
        self.use_wandb = use_wandb
        self.task_type = task_type
        self.strict = strict
        self.as_binary = as_binary
        self.surface_tol = surface_tol

        self.metrics = {
            "Dice": dice,
            "Jaccard": jaccard,
            "Sensitivity": sensitivity,
            "Precision": precision,
            "Volume Similarity": volume_similarity,
            "True Positives": TP,
            "False Positives": FP,
            "False Negatives": FN,
            "Total Positives Ground Truth": total_pos_gt,
            "Total Positives Prediction": total_pos_pred,
        }
        self.obj_metrics = []
        self.surface_metrics = []

        if self.task_type == "segmentation":
            self.metrics = {
                "Dice": dice,
                "Jaccard": jaccard,
                "Sensitivity": sensitivity,
                "Precision": precision,
                "Volume Similarity": volume_similarity,
                "True Positives": TP,
                "False Positives": FP,
                "False Negatives": FN,
                "Total Positives Ground Truth": total_pos_gt,
                "Total Positives Prediction": total_pos_pred,
            }

            if do_object_eval:
                self.name += "_OBJ"
                self.obj_metrics = [
                    "_OBJ Total Objects Prediction",
                    "_OBJ Total Objects Ground Truth",
                    "_OBJ True Positives",
                    "_OBJ False Positives",
                    "_OBJ False Negatives",
                    "_OBJ Mean Volume Prediction",
                    "_OBJ Mean Volume Ground Truth",
                    "_OBJ sensitivity",
                    "_OBJ precision",
                    "_OBJ F1",
                ]

            if do_surface_eval:
                self.name += f"_SURFACE{self.surface_tol}"
                self.surface_metrics = [
                    "Average Surface Distance",
                ]

            self.metrics_included_in_streamtable = [
                "Dice",
                "Sensitivity",
                "Precision",
                "Jaccard",
                "Volume Similarity",
                "_OBJ sensitivity",
                "_OBJ precision",
                "_OBJ F1",
            ]
        elif self.task_type == "classification":
            self.metrics = {
                "Accuracy": accuracy,
                "F1": dice,
                "Sensitivity": sensitivity,
                "Precision": precision,
                # "AUROC": auroc, # only when probabilities are available
            }

            self.metrics_included_in_streamtable = [
                "Accuracy",
                "F1",
                "Sensitivity",
                "Precision",
                # "AUROC", # only when probabilities are available
            ]
        elif self.task_type == "regression":
            raise NotImplementedError
            # self.metrics = {
            #     "MAE": mae,
            #     "RMSE": rmse,
            #     "R2": r2,
            # }
        else:
            raise ValueError(f"Unknown task type {self.task_type}")

        if self.as_binary:
            self.name += "_BINARY"
            self.labels = [0, 1]
            self.labelarr = np.sort(np.array(self.labels, dtype=np.uint8))
        elif isinstance(self.labels, int):
            self.labelarr = np.sort(np.arange(self.labels, dtype=np.uint8))
        elif isinstance(self.labels, list):
            self.labelarr = np.sort(np.array(self.labels, dtype=np.uint8))
        elif isinstance(self.labels, dict):
            self.labelarr = np.sort(np.array(list(self.labels.keys()), dtype=np.uint8))
        else:
            raise ValueError(f"Incorrect label format. Got {type(self.labels)}")

        self.folder_with_predictions = folder_with_predictions
        self.folder_with_ground_truth = folder_with_ground_truth

        self.outpath = join(self.folder_with_predictions, f"{self.name}.json")

        if extension is None:
            if self.task_type == "classification":
                extension = ".txt"
            else:
                extension = "nii.gz"

        self.pred_subjects = subfiles(self.folder_with_predictions, suffix=extension, join=False)
        self.gt_subjects = subfiles(self.folder_with_ground_truth, suffix=extension, join=False)

        print(
            f"\n"
            f"STARTING EVALUATION \n"
            f"Looking for {extension} files in the following directiories\n"
            f"Folder with predictions: {self.folder_with_predictions}\n"
            f"Folder with ground truth: {self.folder_with_ground_truth}\n"
        )

    def sanity_checks(self):
        assert self.pred_subjects <= self.gt_subjects, "Ground Truth is missing for predicted scans"

        assert self.gt_subjects <= self.pred_subjects, "Prediction is missing for Ground Truth of scans"

        # Check if the Ground Truth directory is a subdirectory of a 'TaskXXX_MyTask' folder.
        # If so, there should be a dataset.json where we can double check that the supplied classes
        # match with the expected classes for the dataset.
        gt_is_task = [i for i in self.folder_with_ground_truth.split(os.sep) if "Task" in i]
        if gt_is_task:
            gt_task = gt_is_task[0]
            dataset_json = join(get_raw_data_path(), gt_task, "dataset.json")
            if isfile(dataset_json):
                dataset_json = load_json(dataset_json)
                print(f"Labels found in dataset.json: {list(dataset_json['labels'].keys())}")

    def run(self):
        if isfile(self.outpath) and not self.overwrite:
            print(f"Evaluation file already present in {self.outpath}. Skipping.")
        else:
            if self.strict:
                self.sanity_checks()
            results_dict = self.evaluate_folder()
            self.save_as_json(results_dict)
            if self.use_wandb:
                self.update_streamtable(results_dict)

    def evaluate_folder(self):
        if self.task_type == "classification":
            return evaluate_folder_cls(
                labels=self.labelarr,
                metrics=self.metrics,
                subjects=self.pred_subjects,
                folder_with_predictions=self.folder_with_predictions,
                folder_with_ground_truth=self.folder_with_ground_truth,
            )
        elif self.task_type == "segmentation":
            multilabel = self.regions is not None
            return evaluate_folder_segm(
                labels=self.labels if multilabel else self.labelarr,
                metrics=self.metrics,
                subjects=self.pred_subjects,
                folder_with_predictions=self.folder_with_predictions,
                folder_with_ground_truth=self.folder_with_ground_truth,
                as_binary=self.as_binary,
                obj_metrics=self.obj_metrics,
                surface_metrics=self.surface_metrics,
                surface_tol=self.surface_tol,
                regions=self.regions,
                multilabel=multilabel,
            )
        else:
            raise NotImplementedError("Invalid task type")

    def save_as_json(self, dict):
        print("Saving results.json \n \n ########################################################################")
        with open(self.outpath, "w") as f:
            json.dump(dict, f, default=float, indent=4)

    def update_streamtable(self, results_dict):
        """
        Save evaluation results to a wandb StreamTable

        :param results_dict: dictionary with evaluation results
        """
        from weave.monitoring import StreamTable

        task = self.outpath.split(os.path.sep)[-5]
        target = self.outpath.split(os.path.sep)[-6]
        model_name = "/".join(self.outpath.split(os.path.sep)[-4:])

        st = StreamTable(table_name=task, entity_name=wandb.api.viewer()["entity"], project_name="Yucca")

        stream_dict = {"0. Experiment": model_name, "0. Target Task": target}

        if self.task_type == "classification":
            stream_dict = {**stream_dict, **results_dict}

        elif self.task_type == "segmentation":
            for key, _ in results_dict.items():
                if key == "0":
                    continue
                else:
                    stream_dict.update(
                        {f"{key}. " + k: v for k, v in results_dict[key].items() if k in self.metrics_included_in_streamtable}
                    )

        else:
            raise NotImplementedError("Task type not supported")

        st.log(stream_dict)
        st.finish()
