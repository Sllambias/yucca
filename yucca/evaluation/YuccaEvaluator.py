import os
import numpy as np
import nibabel as nib
import json
import sys
import wandb
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, load_json, isfile
from sklearn.metrics import confusion_matrix
from yucca.evaluation.metrics import (
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
)
from yucca.evaluation.obj_metrics import get_obj_stats_for_label
from yucca.paths import yucca_raw_data
from weave.monitoring import StreamTable
from tqdm import tqdm


class YuccaEvaluator(object):
    def __init__(
        self, labels: list | int, folder_with_predictions, folder_with_ground_truth, do_object_eval=False, as_binary=False
    ):
        self.name = "results"
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

        self.metrics_included_in_streamtable = [
            "Dice",
            "Jaccard",
            "Sensitivity",
            "Precision",
            "Volume Similarity",
            "_OBJ sensitivity",
            "_OBJ precision",
            "_OBJ F1",
        ]

        if isinstance(labels, int):
            self.labels = [str(i) for i in range(labels)]
        else:
            self.labels = labels
        self.as_binary = as_binary
        if self.as_binary:
            self.labels = ["0", "1"]
            self.name += "_BINARY"

        self.labelarr = np.array(self.labels, dtype=np.uint8)
        self.folder_with_predictions = folder_with_predictions
        self.folder_with_ground_truth = folder_with_ground_truth

        self.outpath = join(self.folder_with_predictions, f"{self.name}.json")

        self.pred_subjects = subfiles(self.folder_with_predictions, suffix=".nii.gz", join=False)
        self.gt_subjects = subfiles(self.folder_with_ground_truth, suffix=".nii.gz", join=False)

        print(
            f"\n"
            f"STARTING EVALUATION \n"
            f"Folder with predictions: {self.folder_with_predictions}\n"
            f"Folder with ground truth: {self.folder_with_ground_truth}\n"
            f"Evaluating performance on labels: {self.labels}"
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
            dataset_json = join(yucca_raw_data, gt_task, "dataset.json")
            if isfile(dataset_json):
                dataset_json = load_json(dataset_json)
                print(f"Labels found in dataset.json: {list(dataset_json['labels'].keys())}")

    def run(self):
        if isfile(self.outpath):
            print(f"Evaluation file already present in {self.outpath}. Skipping.")
        else:
            self.sanity_checks()
            dict = self.evaluate_folder()
            self.save_as_json(dict)
            self.update_streamtable(dict["mean"])

    def evaluate_folder(self):
        sys.stdout.flush()
        resultdict = {}
        meandict = {}

        for label in self.labels:
            meandict[label] = {k: [] for k in list(self.metrics.keys()) + self.obj_metrics}

        for case in tqdm(self.pred_subjects, desc="Evaluating"):
            casedict = {}
            predpath = join(self.folder_with_predictions, case)
            gtpath = join(self.folder_with_ground_truth, case)

            pred = nib.load(predpath)
            gt = nib.load(gtpath)
            if self.as_binary:
                cmat = confusion_matrix(
                    np.around(gt.get_fdata().flatten()).astype(bool).astype(int),
                    np.around(pred.get_fdata().flatten()).astype(bool).astype(int),
                    labels=self.labelarr,
                )
            else:
                cmat = confusion_matrix(
                    np.around(gt.get_fdata().flatten()).astype(int),
                    np.around(pred.get_fdata().flatten()).astype(int),
                    labels=self.labelarr,
                )

            for label in self.labelarr:
                labeldict = {}

                tp = cmat[label, label]
                fp = sum(cmat[:, label]) - tp
                fn = sum(cmat[label, :]) - tp
                tn = np.sum(cmat) - tp - fp - fn  # often a redundant and meaningless metric
                for k, v in self.metrics.items():
                    labeldict[k] = round(v(tp, fp, tn, fn), 4)
                    meandict[str(label)][k].append(labeldict[k])

                if self.obj_metrics:
                    # now for the object metrics
                    obj_labeldict = get_obj_stats_for_label(gt, pred, label, as_binary=self.as_binary)
                    for k, v in obj_labeldict.items():
                        labeldict[k] = round(v, 4)
                        meandict[str(label)][k].append(labeldict[k])

                casedict[str(label)] = labeldict
            casedict["Prediction:"] = predpath
            casedict["Ground Truth:"] = gtpath

            resultdict[case] = casedict
        for label in self.labels:
            meandict[label] = {
                k: round(np.nanmean(v), 4) if not np.all(np.isnan(v)) else 0 for k, v in meandict[label].items()
            }
        resultdict["mean"] = meandict

        return resultdict

    def save_as_json(self, dict):
        print(f"Saving results.json" "\n" "\n" f"########################################################################")
        with open(self.outpath, "w") as f:
            json.dump(dict, f, default=float, indent=4)

    def update_streamtable(self, dict):
        task = self.outpath.split(os.path.sep)[-5]
        target = self.outpath.split(os.path.sep)[-6]
        model_name = "/".join(self.outpath.split(os.path.sep)[-4:])

        st = StreamTable(table_name=task, entity_name=wandb.api.viewer()["entity"], project_name="Yucca")

        stream_dict = {"0. Experiment": model_name, "0. Target Task": target}

        for key, _ in dict.items():
            if key == "0":
                continue
            else:
                stream_dict.update(
                    {f"{key}. " + k: v for k, v in dict[key].items() if k in self.metrics_included_in_streamtable}
                )
        st.log(stream_dict)
        st.finish()
