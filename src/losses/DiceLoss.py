import torch.nn.functional as F
import torch
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def one_hot_encode(self, input):
        tensor_list = []
        for i in range(self.n_classes):
            tmp = (input == i) * torch.ones_like(input)
            tensor_list.append(tmp)
        output_tensor = torch.cat(tensor_list, dim = 1)
        return output_tensor.float()
    
    def dice_loss(self, predict, target):
        target    = target.float()
        smooth    = 1e-5
        intersect = torch.sum(predict * target)
        dice      = (2 * intersect + smooth) / (torch.sum(target * target) + torch.sum(predict * predict) + smooth)
        return 1.0 - dice

    def forward(self, input, target):

        target = self.one_hot_encode(target)
        weight = [1] * self.n_classes
        assert input.shape == target.shape, 'size must match'
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            diceloss = self.dice_loss(input[:, i], target[:, i])
            class_wise_dice.append(diceloss)
            loss += diceloss * weight[i]
        loss = loss / self.n_classes
        return loss
        

