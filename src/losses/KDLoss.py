import torch
from torch import nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    def __init__(self, T:float):
        super().__init__()
        self.T = T

    def forward(self,input: torch.tensor,target: torch.tensor):
        assert input.size() == target.size()
        input_log_softmax = F.log_softmax(input, dim = 1)/self.T
        target_softmax    = F.softmax(target, dim = 1)/self.T
        kl_div            = F.kl_div(input_log_softmax, target_softmax, reduction = 'none')
        kl_div            = kl_div.mean()
        return kl_div