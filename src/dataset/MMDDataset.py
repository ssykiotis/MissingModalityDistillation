import torch
import logging
import numpy as np
from ..mm_dataclasses import *
from omegaconf import DictConfig


class MissingModalityDistillationDataset:
    
    def __init__(self, config:DictConfig, x: np.ndarray, y:np.ndarray, training_mode:str, norm_params: NormParams):
        self.training_mode    = training_mode
        self.missing_modalies = self.config.dataset.missing_modalities
        self.config           = config
        self.x = x
        self.y = y

        self.normalization = self.config.dataset.normalization
        self.norm_params   = self.set_norm_params(norm_params)

        self.filter_data()
    
    def set_norm_params(self,norm_params:NormParams):
        if norm_params.x_min is None:
            norm_params.x_min = self.x.min(axis = (0,2,3))
            norm_params.x_max = self.x.max(axis = (0,2,3))
        
        if norm_params.x_mean is None:
            norm_params.x_mean = self.x.mean(axis = (0,2,3))
            norm_params.x_std  = self.x.std(axis = (0,2,3))
        
        return norm_params

    #TODO
    def __getitem__(self,index):
        # prepei na bei kai ena clause gia to distillation, na gyrnaei 2 zeygaria
        if self.training_mode != 'distillation':
            x = self.x[index]
            y = self.y[index]

            x = self.normalize(x)
            x = torch.tensor(x).contiguous()
            y = torch.tensor(y).contiguous()
            return x, y
        else:
            x = self.x[index]
            y = self.y[index]
            modalities_to_keep = [i for i in range(x.shape[0]) if i not in self.missing_modalies]

            x_missing = x[modalities_to_keep,:,:]

            return x, x_missing, y

    def normalize(self,x):
        if self.normalization == 'minmax':
            return ((x.T-self.norm_params.x_min)/(self.norm_params.x_max-self.norm_params.x_min)).T 
        elif self.normalization == 'gauss':
            return ((x.T - self.norm_params.x_mean)/self.norm_params.x_std).T
        else:
            return x
    
    #TODO
    def filter_data(self):
        if self.training_mode in ['teacher','distillation']:
            pass
        elif self.training_mode == 'student':
            modalities_to_keep = [i for i in range(self.x.shape[1]) if i not in self.missing_modalies]
            self.x = self.x[:,modalities_to_keep,:,:]
        pass


    
    #TODO
    def __len__(self):
        return(self.x.shape[0])