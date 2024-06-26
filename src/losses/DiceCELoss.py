from .DiceLoss import DiceLoss
from .CrossEntropyLoss import *

class DiceCeLoss(nn.Module):
    # predict : output of model (i.e. no softmax)[N,C,*]
    # target : gt of img [N,1,*]
    def __init__(self, n_classes, alpha = 1.0, weighted = True):
        '''
        calculate loss:
            celoss + alpha*celoss
            alpha : default is 1
        '''
        super().__init__()
        self.alpha       = alpha
        self.n_classes   = n_classes
        self.diceloss    = DiceLoss(self.n_classes)
        self.celoss      = WeightedCrossEntropyLoss(self.n_classes) if weighted else RobustCrossEntropyLoss()

    def forward(self, predict, label):
        # predict is output of the model, i.e. without softmax [N,C,*]
        # label is not one hot encoding [N,1,*]

        diceloss = self.diceloss(predict, label)
        celoss   = self.celoss(  predict, label)
        loss     = (1-self.alpha) * celoss + self.alpha * diceloss
        return diceloss, celoss, loss
