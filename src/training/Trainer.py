import logging
import time
import numpy as np
import hydra

from omegaconf import DictConfig
from src.dataset.DatasetParser import *
import torch.utils.data as data_utils

class Trainer:

    def __init__(self,config:DictConfig, ds_parser: GeneralDatasetParser):

        self.model = hydra.utils.instantiate(config.model)


        self.epochs     = config.epochs
        self.initial_lr = config.lr
        self.loss_fn    = hydra.utils.instantiate(config.loss)
        self.optimizer  = None

        self.train_dl = self.get_dataloader('train')
        self.val_dl   = self.get_dataloader('val')
        
        pass

    #TODO
    def train(self):
        for epoch in range(self.epochs):
            curr_lr = self.intial_lr * (1.0-np.float32(epoch)/np.float32(self.epochs))**(0.9)
            self.train_one_epoch(curr_lr)

        pass
    #TODO
    def train_one_epoch(self,lr):
        self.model.train()
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
        test_dl = self.get_dataloader('test')
        pass

    def get_dataloader(self,mode: str):
        dataset = self.ds_parser.get_dataset(mode)
        self.norm_params = dataset.norm_params

        if mode == "train":

            dataloader = data_utils.Dataloader(dataset,
                                               batch_size = self.config.batch_size,
                                               shuffle    = True,
                                               pin_memory = True,
                                               drop_last  = False
                                               )
        else:
            dataloader = data_utils.Dataloader(dataset,
                                        batch_size = self.config.batch_size,
                                        shuffle    = False,
                                        pin_memory = True,
                                        drop_last  = False
                                        )
        return dataloader
