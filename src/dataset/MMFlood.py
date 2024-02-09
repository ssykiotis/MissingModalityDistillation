from omegaconf import DictConfig
from .MMDDataset import MissingModalityDistillationDataset
from ..mm_dataclasses import *
from .DatasetParser   import GeneralDatasetParser

import os
import numpy as np
import rasterio
import logging

class MMFloodParser(GeneralDatasetParser):

    def __init__(self, params:DictConfig):
        # self.params        = params
        self.data_location = params.data_location
        self.val_per       = params.val_per
        self.sample_paths  = self.parse_data()

        self.img_dim            = params.img_dim
        # self.normalization      = params.normalization
        self.missing_modalities = params.missing_modalities

        self.train_samples, self.val_samples, self.test_samples = self.train_test_split()

        self.norm_params   = NormParams(normalization = params.normalization)


    def parse_data(self,shuffle:bool = False) -> list:
        """
        parse dataset directory and return paths to sample folders. optionally shuffles samples
        """
        paths = [f'{self.data_location}/{f}' for f in sorted(os.listdir(self.data_location))]
        if shuffle:
            paths.shuffle() #TODO: check if call is correct
        return paths   

    def train_test_split(self) -> tuple[list, list, list]:
        num_samples = len(self.sample_paths)
        val_end     = int(num_samples * self.val_per)
        train_end   = int(val_end * self.val_per)

        train_paths = self.sample_paths[:train_end]
        val_paths   = self.sample_paths[train_end:val_end]
        test_paths  = self.sample_paths[val_end:]
        
        return train_paths, val_paths, test_paths
        

    def get_dataset(self, dataset_type:str, training_mode:str) -> MissingModalityDistillationDataset:
        
        x, y = self.read_data(dataset_type)
        
        if dataset_type == 'train':
            if self.norm_params.normalization == 'gauss':
                self.norm_params.x_mean = x.astype(np.float32).mean(axis = (0,2,3))
                self.norm_params.x_std  = x.astype(np.float32).std(axis = (0,2,3))
            elif self.norm_params.normalization == 'minmax':
                self.norm_params.x_min = x.astype(np.float32).min(axis = (0,2,3))
                self.norm_params.x_max = x.astype(np.float32).max(axis = (0,2,3))
        logging.info(f'Dataset type: {dataset_type}')
        return MissingModalityDistillationDataset(x, y, training_mode, self.norm_params, self.missing_modalities)

        
    def read_data(self, dataset_type:str) -> [np.ndarray, np.ndarray]:
        s1_paths, dem_paths, mask_paths = [], [], []

        paths = self.train_samples if  dataset_type=='train' else self.val_samples if dataset_type=='val' else self.test_samples


        for item in paths:
            s1_paths.extend([f'{item}/s1_raw/{p}' for p in sorted(os.listdir(f'{item}/s1_raw'))])
            dem_paths.extend([f'{item}/DEM/{p}'   for p in sorted(os.listdir(f'{item}/DEM'))])
            mask_paths.extend([f'{item}/mask/{p}' for p in sorted(os.listdir(f'{item}/mask'))])

        assert len(s1_paths)==len(dem_paths), 'Number of s1 patches and DEM patches missmatch!'
        assert len(s1_paths)==len(mask_paths),'Number of s1 patches and mask patches missmatch!'

        patches = list(zip(s1_paths, dem_paths, mask_paths))

        num_patches = len(patches)

        x = np.zeros((num_patches, 3,  self.img_dim, self.img_dim), dtype = np.float32)
        y = np.zeros((num_patches, 1 , self.img_dim, self.img_dim), dtype = np.int8)

        for idx,patch in enumerate(patches):
            p_s1, p_dem, p_mask = patch
            img = rasterio.open(p_s1  ).read()
            dem = rasterio.open(p_dem ).read()
            m   = rasterio.open(p_mask).read()

            x[idx, :2, :, :] = img
            x[idx,  2, :, :] = dem

            y[idx] = m

        return x, y

