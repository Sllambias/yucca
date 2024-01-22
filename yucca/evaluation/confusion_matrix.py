import torch
from torchmetrics.classification import ConfusionMatrix


def torch_confusion_matrix_from_logits(pred, seg):
    if len(seg.shape) == len(pred.shape):
        assert seg.shape[1] == 1
        seg = seg[:, 0]

    n_classes = pred.shape[1]

    cfm = ConfusionMatrix(task="multiclass", num_classes=n_classes)

    if torch.cuda.is_available():
        cfm = cfm.to("cuda")
    matrix = cfm(pred, seg)
    return matrix


def torch_get_tp_fp_tn_fn(confusion_matrix, ignore_label=0):
    TP = []
    FP = []
    TN = []
    FN = []

    for label in range(confusion_matrix.shape[0]):
        if label == ignore_label:
            continue
        tp = confusion_matrix[label, label]
        fp = torch.sum(confusion_matrix[:, label]) - tp
        fn = torch.sum(confusion_matrix[label]) - tp
        tn = torch.sum(confusion_matrix) - tp - fp - fn
        TP.append(tp.cpu().numpy())
        FP.append(fp.cpu().numpy())
        TN.append(tn.cpu().numpy())
        FN.append(fn.cpu().numpy())
    return TP, FP, TN, FN
