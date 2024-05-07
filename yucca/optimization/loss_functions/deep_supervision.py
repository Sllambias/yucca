from torch import nn
import numpy as np


class DeepSupervisionLoss(nn.Module):
    """
    Takes in a list of logits and a list of downsampled segmentations.
    Expects the first entry in each list to be the full-size/original segmentation, which will
    also be the one with the highest weight
    """

    def __init__(self, loss, weights=None):
        super().__init__()
        self.loss = loss
        self.weights = weights

    def forward(self, list_of_logits, list_of_segs):
        if self.weights is None:
            weights = np.array([1 / (2**i) for i in range(len(list_of_logits))])
            self.weights = weights / np.sum(weights)
        loss = self.loss(list_of_logits[0], list_of_segs[0]) * self.weights[0]
        for i in range(1, len(list_of_logits)):
            loss += self.loss(list_of_logits[i], list_of_segs[i]) * self.weights[i]
        return loss
