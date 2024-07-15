import sys
import numpy as np
import nibabel as nib
from typing import Optional
from yucca.functional.transforms.label_transforms import convert_labels_to_regions
from yucca.functional.utils.nib_utils import get_nib_spacing
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import join
from sklearn.metrics import confusion_matrix


def evaluate_multilabel_folder_segm(
    labels,
    metrics,
    subjects,
    folder_with_predictions,
    folder_with_ground_truth,
    as_binary: Optional[bool] = False,
    obj_metrics: Optional[bool] = False,
    regions: Optional[list] = None,
    surface_metrics: Optional[bool] = False,
):
    # predictions are multilabel at this point (c, h, w, d)
    # ground truth MAY be converted, but may also be (h, w, d) in which case we use the label_regions
    # to convert for multilabel evaluation

    sys.stdout.flush()
    resultdict = {}
    meandict = {}

    for label in labels:
        meandict[str(label)] = {k: [] for k in list(metrics.keys()) + obj_metrics + surface_metrics}

    for case in tqdm(subjects, desc="Evaluating"):
        casedict = {}
        predpath = join(folder_with_predictions, case)
        gtpath = join(folder_with_ground_truth, case)

        pred = nib.load(predpath)
        spacing = get_nib_spacing(pred)
        pred = pred.get_fdata()
        gt = nib.load(gtpath).get_fdata()

        if len(pred.shape) == len(gt.shape) + 1:
            # In thise case gt has not been converted to multilabel
            assert (
                regions is not None
            ), "Regions must be supplied if ground truth is not already multilabel (i.e. multiple channels)"
            gt = convert_labels_to_regions(gt[np.newaxis], regions)

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
            labeldict = {}

            tp = cmat[label, label]
            fp = sum(cmat[:, label]) - tp
            fn = sum(cmat[label, :]) - tp
            tn = np.sum(cmat) - tp - fp - fn  # often a redundant and meaningless metric
            for k, v in metrics.items():
                labeldict[k] = round(v(tp, fp, tn, fn), 4)
                meandict[str(label)][k].append(labeldict[k])

            if obj_metrics:
                # now for the object metrics
                obj_labeldict = get_obj_stats_for_label(gt, pred, label, as_binary=as_binary)
                for k, v in obj_labeldict.items():
                    labeldict[k] = round(v, 4)
                    meandict[str(label)][k].append(labeldict[k])

            if surface_metrics:
                surface_labeldict = get_surface_metrics_for_label(gt, pred, label, as_binary=as_binary)
                for k, v in surface_labeldict.items():
                    labeldict[k] = round(v, 4)
                    meandict[str(label)][k].append(labeldict[k])
            casedict[str(label)] = labeldict
        casedict["Prediction:"] = predpath
        casedict["Ground Truth:"] = gtpath

        resultdict[case] = casedict
        del pred, gt, cmat

    for label in labels:
        meandict[str(label)] = {
            k: round(np.nanmean(v), 4) if not np.all(np.isnan(v)) else 0 for k, v in meandict[str(label)].items()
        }
    resultdict["mean"] = meandict
    return resultdict
