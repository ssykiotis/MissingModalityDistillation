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
        




class DiceCeLoss(nn.Module):
    # predict : output of model (i.e. no softmax)[N,C,*]
    # target : gt of img [N,1,*]
    def __init__(self, num_classes, alpha = 1.0, weighted = True):
        '''
        calculate loss:
            celoss + alpha*celoss
            alpha : default is 1
        '''
        super().__init__()
        self.alpha       = alpha
        self.num_classes = num_classes
        self.diceloss    = DiceLoss(self.num_classes)
        self.celoss      = WeightedCrossEntropyLoss(self.num_classes) if weighted else RobustCrossEntropyLoss()

    def forward(self, predict, label):
        # predict is output of the model, i.e. without softmax [N,C,*]
        # label is not one hot encoding [N,1,*]

        diceloss = self.diceloss(predict, label)
        celoss   = self.celoss(predict, label)
        loss     = celoss + self.alpha * diceloss
        return diceloss, celoss, loss

class KDLoss:
    def __init__(self,):
        pass

    def forward(self,input, target):
        pass


class PrototypeLoss:
    def __init__(self,):
        pass

    def forward(self,input, target):
        pass