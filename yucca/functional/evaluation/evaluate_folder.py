import sys
import numpy as np
import nibabel as nib
import logging
from typing import Optional
from yucca.functional.transforms.label_transforms import convert_labels_to_regions, translate_region_labels
from yucca.functional.utils.nib_utils import get_nib_spacing
from yucca.functional.utils.loading import read_file_to_nifti_or_np
from yucca.functional.evaluation.obj_metrics import get_obj_stats_for_label
from yucca.functional.evaluation.surface_metrics import get_surface_metrics_for_label
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import join
from sklearn.metrics import confusion_matrix
from yucca.functional.evaluation.metrics import auroc


def evaluate_folder_segm(
    folder_with_predictions: str,
    folder_with_ground_truth: str,
    labels: dict,
    metrics: dict,
    subjects: list,
    as_binary: Optional[bool] = False,
    multilabel: bool = False,
    obj_metrics: Optional[bool] = False,
    regions: Optional[dict] = None,
    surface_metrics: Optional[bool] = False,
    surface_tol: int = 1,
):
    sys.stdout.flush()
    result_dict = {}
    mean_dict = {}

    if multilabel:
        assert regions is not None
        # predictions are multilabel at this point (c, h, w, d)
        # ground truth MAY be converted, but may also be (h, w, d) in which case we use the label_regions
        # to convert for multilabel evaluation

        labels_from_regions = np.array(range(len(regions.keys()) + 1), dtype=np.uint8)
        logging.info(f"Multilabel segmentation evaluation with regions: {regions} and labels: {labels}")
        metric_labels = regions.keys()
    else:
        logging.info(f"segmentation evaluation with labels: {labels}")
        metric_labels = labels

    for label in metric_labels:
        mean_dict[str(label)] = {k: [] for k in list(metrics.keys()) + obj_metrics + surface_metrics}

    evaluation_args = {
        "folder_with_ground_truth": folder_with_ground_truth,
        "folder_with_predictions": folder_with_predictions,
        "labels": labels,
        "as_binary": as_binary,
        "obj_metrics": obj_metrics,
        "surface_metrics": surface_metrics,
        "surface_tol": surface_tol,
        "metrics": metrics,
    }

    for case in tqdm(subjects, desc="Evaluating"):
        if multilabel:
            case_dict = evaluate_multilabel_case_segm(
                case, regions=regions, labels_from_regions=labels_from_regions, **evaluation_args
            )
        else:
            case_dict = evaluate_case_segm(case, **evaluation_args)

        result_dict[case] = case_dict

        for label in metric_labels:
            for metric, val in case_dict[str(label)].items():
                mean_dict[str(label)][metric].append(val)

    for label in metric_labels:
        mean_dict[str(label)] = {
            k: round(np.nanmean(v), 4) if not np.all(np.isnan(v)) else 0 for k, v in mean_dict[str(label)].items()
        }
    result_dict["mean"] = mean_dict

    return result_dict


def evaluate_multilabel_case_segm(
    case: str,
    folder_with_predictions: str,
    folder_with_ground_truth: str,
    labels: dict,
    labels_from_regions: np.array,
    metrics: dict,
    regions: dict,
    as_binary: Optional[bool] = False,
    obj_metrics: Optional[bool] = False,
    surface_metrics: Optional[bool] = False,
    surface_tol: int = 1,
):
    assert regions is not None

    case_dict = {}
    predpath = join(folder_with_predictions, case)
    gtpath = join(folder_with_ground_truth, case)
    case_dict["prediction_path"] = predpath
    case_dict["ground_truth_path"] = gtpath

    pred = nib.load(predpath)
    spacing = get_nib_spacing(pred)[:3]
    pred = pred.get_fdata().astype(np.uint8)
    pred = pred.transpose([3, 0, 1, 2])
    gt = nib.load(gtpath).get_fdata()

    if len(pred.shape) == len(gt.shape) + 1:
        # In thise case gt has not been converted to multilabel
        assert (
            regions is not None
        ), "Regions must be supplied if ground truth is not already multilabel (i.e. multiple channels)"
        translated_regions = translate_region_labels(regions=regions, labels=labels)
        gt = convert_labels_to_regions(gt[np.newaxis], translated_regions)
        for i in range(len(regions.keys())):
            pred[i] *= 1 + i
            gt[i] *= 1 + i

    if as_binary:
        cmat = confusion_matrix(
            np.around(gt.flatten()).astype(bool).astype(np.uint8),
            np.around(pred.flatten()).astype(bool).astype(np.uint8),
            labels=labels_from_regions,
        )
    else:
        cmat = confusion_matrix(
            np.around(gt.flatten()).astype(np.uint8),
            np.around(pred.flatten()).astype(np.uint8),
            labels=labels_from_regions,
        )

    for label, region_name in enumerate(regions.keys()):
        label += 1
        labeldict = {}

        tp = cmat[label, label]
        fp = sum(cmat[:, label]) - tp
        fn = sum(cmat[label, :]) - tp
        tn = np.sum(cmat) - tp - fp - fn  # often a redundant and meaningless metric
        for k, v in metrics.items():
            labeldict[k] = round(v(tp, fp, tn, fn), 4)

        if obj_metrics:
            raise NotImplementedError
            # now for the object metrics
            # obj_labeldict = get_obj_stats_for_label(gt, pred, label, spacing=spacing, as_binary=as_binary)
            # for k, v in obj_labeldict.items():
            #    labeldict[k] = round(v, 4)

        if surface_metrics:
            if label == 0:
                surface_labeldict = get_surface_metrics_for_label(
                    gt[label], pred[label], 0, spacing=spacing, tol=surface_tol, as_binary=as_binary
                )
            else:
                surface_labeldict = get_surface_metrics_for_label(
                    gt[label - 1], pred[label - 1], label, spacing=spacing, tol=surface_tol, as_binary=as_binary
                )
            for k, v in surface_labeldict.items():
                labeldict[k] = round(v, 4)

        case_dict[str(region_name)] = labeldict

    # Case dict contains for labels "0", ... and metrics "dice", "f1", ...
    # {
    #    "0": { "dice": 0.1, "f1": 0.2, ... },
    #    ...
    # }
    return case_dict


def evaluate_case_segm(
    case: str,
    folder_with_predictions: str,
    folder_with_ground_truth: str,
    labels: dict,
    metrics: dict,
    as_binary: Optional[bool] = False,
    obj_metrics: Optional[bool] = False,
    surface_metrics: Optional[bool] = False,
    surface_tol: int = 1,
):
    case_dict = {}
    predpath = join(folder_with_predictions, case)
    gtpath = join(folder_with_ground_truth, case)

    case_dict["prediction_path"] = predpath
    case_dict["ground_truth_path"] = gtpath

    pred = read_file_to_nifti_or_np(predpath)
    if isinstance(pred, nib.Nifti1Image):
        spacing = get_nib_spacing(pred)
        pred = pred.get_fdata()
    else:
        spacing = [1.0] * len(pred.shape)

    gt = read_file_to_nifti_or_np(gtpath)
    if isinstance(gt, nib.Nifti1Image):
        gt = gt.get_fdata()

    if as_binary:
        cmat = confusion_matrix(
            np.around(gt.flatten()).astype(bool).astype(np.uint8),
            np.around(pred.flatten()).astype(bool).astype(np.uint8),
            labels=labels,
        )
    else:
        cmat = confusion_matrix(
            np.around(gt.flatten()).astype(np.uint8),
            np.around(pred.flatten()).astype(np.uint8),
            labels=labels,
        )

    for label in labels:
        label_dict = {}

        tp = cmat[label, label]
        fp = sum(cmat[:, label]) - tp
        fn = sum(cmat[label, :]) - tp
        tn = np.sum(cmat) - tp - fp - fn  # often a redundant and meaningless metric
        for k, v in metrics.items():
            label_dict[k] = round(v(tp, fp, tn, fn), 4)

        if obj_metrics:
            # now for the object metrics
            obj_labeldict = get_obj_stats_for_label(gt, pred, label, spacing=spacing, as_binary=as_binary)
            for k, v in obj_labeldict.items():
                label_dict[k] = round(v, 4)

        if surface_metrics:
            surface_labeldict = get_surface_metrics_for_label(
                gt, pred, label, spacing=spacing, tol=surface_tol, as_binary=as_binary
            )
            for k, v in surface_labeldict.items():
                label_dict[k] = round(v, 4)

        case_dict[str(label)] = label_dict

    # Case dict contains for labels "0", ... and metrics "dice", "f1", ...
    # {
    #    "0": { "dice": 0.1, "f1": 0.2, ... },
    #    ...
    # }
    return case_dict


def evaluate_folder_cls(
    labels,
    metrics,
    subjects,
    folder_with_predictions,
    folder_with_ground_truth,
):
    """
    Evaluate classification results
    """
    sys.stdout.flush()
    resultdict = {}

    predictions = []
    prediction_probs = []
    ground_truths = []

    # Flag to check if we have prediction probabilities to calculate AUROC
    use_probs = False

    # load predictions and ground truths
    for case in tqdm(subjects, desc="Evaluating"):
        predpath = join(folder_with_predictions, case)
        gtpath = join(folder_with_ground_truth, case)

        pred: int = np.loadtxt(predpath)
        gt: int = np.loadtxt(gtpath)

        try:
            if len(prediction_probs) == 0:
                print("Prediction probabilities found. Will use them for evaluation.")
                use_probs = True

            pred_probs = np.load(predpath.replace(".txt", ".npz"))["data"]  # contains output probabilities
            prediction_probs.append(pred_probs)
        except FileNotFoundError:
            pred_probs = None

        predictions.append(pred)
        ground_truths.append(gt)

    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    # calculate per-class metrics
    cmat = confusion_matrix(ground_truths, predictions, labels=labels)

    resultdict["per_class"] = {}

    for label in labels:
        tp = cmat[label, label]
        fp = sum(cmat[:, label]) - tp
        fn = sum(cmat[label, :]) - tp
        tn = np.sum(cmat) - tp - fp - fn

        labeldict = {}

        for k, v in metrics.items():
            labeldict[k] = round(v(tp, fp, tn, fn), 4)

        resultdict["per_class"][str(label)] = labeldict

    # calculate AUROC
    if use_probs:
        auroc_per_class: list[float] = auroc(ground_truths, prediction_probs)
        for label, score in zip(labels, auroc_per_class):
            resultdict["per_class"][str(label)]["AUROC"] = round(score, 4)

    # caclulate global (mean) metrics
    resultdict["mean"] = {}
    for k, _ in resultdict["per_class"][str(labels[0])].items():
        resultdict["mean"][k] = sum([resultdict["per_class"][str(label)][k] for label in labels])
        resultdict["mean"][k] = round(resultdict["mean"][k] / len(labels), 4)

    return resultdict
