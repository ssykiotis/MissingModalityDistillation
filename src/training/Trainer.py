import logging
import time
import numpy as np

class Trainer:

    def __init__(self,config):

        self.model = hydra.utils.instantiate(config.model)


        self.epochs     = config.epochs
        self.initial_lr = config.lr
        self.loss_fn   = None
        self.optimizer = None
        
        pass

    #TODO
    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            curr_lr = self.intial_lr * (1.0-np.float32(epoch)/np.float32(self.epochs))**(0.9)
            self.train_one_epoch(curr_lr)

        pass
    #TODO
    def train_one_epoch(self,lr):
        for parm in self.optimizer.param_groups:
            parm['lr'] = lr
        loss_values = []
        time1 = time.time()
        pass

    #TODO
    def validate():
        pass

    #TODO
    def test():
        pass