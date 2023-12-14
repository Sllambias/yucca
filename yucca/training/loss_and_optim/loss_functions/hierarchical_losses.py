import torch
import torch.nn.functional as F
from torch import nn, Tensor
from yucca.training.loss_and_optim.utils.LossTree import LossTree
from yucca.training.loss_and_optim.loss_functions.nnUNet_losses import DiceCE
from yucca.training.loss_and_optim.loss_functions.NLL import NLL


class HierarchicalLoss(nn.Module):
    def __init__(self, hierarchical_kwargs={}):
        super(HierarchicalLoss, self).__init__()
        self.root = LossTree(hierarchical_kwargs["rootdict"])  # root represents your hierarchy
        self.loss = NLL(log=True)

    def calculate_loss(self, running_loss, summed_probabilities, target):
        summed_probabilities /= torch.sum(summed_probabilities, axis=1).unsqueeze(1)
        level_loss = self.loss(summed_probabilities, target)
        return running_loss + level_loss

    def get_loss(self, running_loss, summed_probabilities, target, only_last=False):
        if only_last:
            if self.n_level + 1 == len(self.root.loss_layers):
                return self.calculate_loss(running_loss, summed_probabilities, target)
            else:
                return 0
        return self.calculate_loss(running_loss, summed_probabilities, target)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        probabilities = F.softmax(logits, dim=1)
        loss = 0
        precomputed_hierarchy_list = self.root.loss_layers
        for n_level, level_loss_list in enumerate(precomputed_hierarchy_list):
            self.n_level = n_level
            probabilities_tosum = probabilities.clone()
            summed_probabilities = probabilities_tosum
            for branch in level_loss_list:
                # Extract the relevant probabilities according to a branch in our hierarchy.
                branch_probs = torch.FloatTensor()
                branch_probs = branch_probs.to(logits.device)
                for channel in branch:
                    branch_probs = torch.cat((branch_probs, probabilities_tosum[:, channel].unsqueeze(1)), 1)

                # Sum these probabilities into a single slice; this is hierarchical inference.
                summed_tree_branch_slice = branch_probs.sum(1, keepdim=True)

                # Insert inferred probability slice into each channel of summed_probabilities given by branch.
                # This duplicates probabilities for easy passing to standard loss functions such as nll_loss.
                for channel in branch:
                    summed_probabilities[:, channel : (channel + 1), :, :] = summed_tree_branch_slice
            loss = self.get_loss(loss, summed_probabilities, target)
        return loss


class WeightedHierarchicalLoss(HierarchicalLoss):
    def get_loss(self, running_loss, summed_probabilities, target):
        return super().get_loss(running_loss, summed_probabilities, target) * (1 / (self.n_level + 1))


class ConditionalHierarchicalLoss(HierarchicalLoss):
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        eps = 1e-4
        last_summed_probabilities = torch.Tensor()
        previous_branch_for_channel = {}
        probabilities = F.softmax(logits, dim=1)
        loss = 0
        precomputed_hierarchy_list = self.root.loss_layers
        for n_level, level_loss_list in enumerate(precomputed_hierarchy_list):
            self.n_level = n_level
            probabilities_tosum = probabilities.clone()
            summed_probabilities = probabilities_tosum
            for branch in level_loss_list:
                branch_probs = torch.FloatTensor()
                branch_probs = branch_probs.to(logits.device)
                for channel in branch:
                    branch_probs = torch.cat((branch_probs, probabilities_tosum[:, channel].unsqueeze(1)), 1)

                summed_tree_branch_slice = branch_probs.sum(1, keepdim=True)

                for channel in branch:
                    if channel not in previous_branch_for_channel:
                        # First iteration
                        previous_branch_for_channel[channel] = branch

                    if previous_branch_for_channel[channel] != branch:
                        summed_probabilities[:, channel : (channel + 1), :, :] = summed_tree_branch_slice * (
                            last_summed_probabilities[:, channel : (channel + 1), :, :] + eps
                        )
                        previous_branch_for_channel[channel] = branch
                    else:
                        if len(last_summed_probabilities) > 0:
                            summed_probabilities[:, channel : (channel + 1), :, :] = last_summed_probabilities[
                                :, channel : (channel + 1), :, :
                            ]
                        else:
                            summed_probabilities[:, channel : (channel + 1), :, :] = summed_tree_branch_slice

            loss = self.get_loss(loss, summed_probabilities, target, only_last=True)
            last_summed_probabilities = summed_probabilities
        return loss


class HierarchicalDiceCELoss(HierarchicalLoss):
    def __init__(self, hierarchical_kwargs={}):
        super(HierarchicalDiceCELoss, self).__init__(hierarchical_kwargs)
        self.loss_fn_kwargs = {"soft_dice_kwargs": {"apply_softmax": False}}
        self.loss = DiceCE(**self.loss_fn_kwargs)


class WeightedHierarchicalDiceCELoss(HierarchicalDiceCELoss, WeightedHierarchicalLoss):
    def __init__(self, hierarchical_kwargs={}):
        super().__init__(hierarchical_kwargs)


class ConditionalHierarchicalDiceCELoss(HierarchicalDiceCELoss, ConditionalHierarchicalLoss):
    def __init__(self, hierarchical_kwargs={}):
        super().__init__(hierarchical_kwargs)
