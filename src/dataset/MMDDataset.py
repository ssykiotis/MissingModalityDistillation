import torch
import logging
import numpy as np
from ..mm_dataclasses import *
from omegaconf import DictConfig



class MissingModalityDistillationDataset:
    
    def __init__(self, x: np.ndarray, y:np.ndarray, training_mode:str, norm_params: NormParams, missing_modalities:list = None):
        
        self.training_mode = training_mode

        self.x = x
        self.y = y

        if missing_modalities:
            self.modalities_to_keep = [i for i in range(self.x.shape[1]) if i not in missing_modalities]

        self.norm_params   = self.set_norm_params(norm_params)
        self.filter_data()
        self.log_norm_params()

    def log_norm_params(self):
        logging.info(f'Dataset Mode: {self.training_mode}')
        logging.info(f'Dataset Size: {self.x.shape[0]}')
        logging.info(f'Normalization Type: {self.norm_params.normalization}')
        logging.info(f'x_min:  {self.norm_params.x_min}')
        logging.info(f'x_max:  {self.norm_params.x_max}')
        logging.info(f'x_mean: {self.norm_params.x_mean}')
        logging.info(f'x_std:  {self.norm_params.x_std}')
    
    def set_norm_params(self,norm_params:NormParams) -> NormParams:
        if norm_params.x_min is None:
            norm_params.x_min = self.x.min(axis = (0,2,3))
            norm_params.x_max = self.x.max(axis = (0,2,3))
        
        if norm_params.x_mean is None:
            norm_params.x_mean = self.x.mean(axis = (0,2,3))
            norm_params.x_std  = self.x.std( axis = (0,2,3))
        
        return norm_params

    def __getitem__(self,index) -> tuple[torch.tensor,torch.tensor] | tuple[torch.tensor,torch.tensor,torch.tensor]:
        x = self.x[index]
        y = self.y[index]
        x = self.normalize(x)

        if self.training_mode == 'baseline':
            x = x[self.modalities_to_keep,:,:]
        elif self.training_mode == 'distillation':
            x_missing = x[self.modalities_to_keep,:,:]
            x_missing = torch.tensor(x_missing).contiguous()

        x = torch.tensor(x).contiguous()
        y = torch.tensor(y).contiguous()

        if self.training_mode != 'distillation':
            return x, y
        else:
            return x, x_missing, y

        # if self.training_mode == 'teacher':
        #     x = torch.tensor(x).contiguous()
        #     y = torch.tensor(y).contiguous()
        #     return x, y
        # elif self.training_mode == 'baseline':

        #     x = x[self.modalities_to_keep,:,:]
        #     x         = torch.tensor(x).contiguous()
        #     y         = torch.tensor(y).contiguous()
        #     return x, y
        # else:
        #     x = self.x[index]
        #     y = self.y[index]

        #     x         = self.normalize(x)
        #     x_missing = x[self.modalities_to_keep,:,:]

        #     x         = torch.tensor(x).contiguous()
        #     y         = torch.tensor(y).contiguous()
        #     x_missing = torch.tensor(x_missing).contiguous()

        #     return x, x_missing, y

    def normalize(self,x) -> np.ndarray:
        if self.norm_params.normalization == 'minmax':
            return ((x.T-self.norm_params.x_min)/(self.norm_params.x_max-self.norm_params.x_min)).T 
        elif self.norm_params.normalization == 'gauss':
            return ((x.T - self.norm_params.x_mean)/self.norm_params.x_std).T
        else:
            return x
    
    def filter_data(self) -> None:
        if self.training_mode in ['teacher','distillation']:
            pass
        elif self.training_mode == 'student':
            self.x = self.x[:,self.modalities_to_keep,:,:]
            if self.norm_params.normalization == 'minmax':
                self.norm_params.x_min  = self.norm_params.x_min[self.modalities_to_keep]
                self.norm_params.x_max  = self.norm_params.x_max[self.modalities_to_keep]
            if self.norm_params.normalization == 'gauss':
                self.norm_params.x_mean = self.norm_params.x_mean[self.modalities_to_keep]
                self.norm_params.x_std  = self.norm_params.x_std[ self.modalities_to_keep]
        pass


    def __len__(self) -> int:
        return(self.x.shape[0])