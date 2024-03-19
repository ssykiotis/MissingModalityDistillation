from torch import nn
import torch
import torch.nn.functional as F

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: torch.tensor, target: torch.tensor) -> torch.tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, n_classes: int, eps = 1e-4):
        super().__init__()
        self.n_classes = n_classes
        self.eps         = eps

    def forward(self, input:torch.tensor, target:torch.tensor)->torch.tensor:
        weight = []
        for c in range(self.n_classes):
            weight_c = torch.sum(target == c).float()
            weight.append(weight_c)

        weight = torch.tensor(weight).to(target.device)
        weight = 1 - weight / (torch.sum(weight)) + 1e-9

        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        # wce_loss_mean = F.cross_entropy(input, target.long(), weight, reduction = 'mean') #TODO: TESTING
        wce_loss_none = F.cross_entropy(input, target.long(), weight, reduction = 'none')
        wce_loss = 0
        for i in range(self.n_classes):
            mask = target == i
            wce_loss += 1/weight[i]*wce_loss_none[mask].sum()/mask.sum()
        return wce_loss




