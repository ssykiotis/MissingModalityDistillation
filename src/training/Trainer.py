import logging
import time
import numpy as np
import hydra
import torch
import os

from omegaconf import DictConfig
from src.dataset.DatasetParser import *
import torch.utils.data as data_utils
import logging
from torchmetrics.classification import Dice
from torch.utils.tensorboard import SummaryWriter 
from ptflops import get_model_complexity_info


class Trainer:

    def __init__(self,config:DictConfig, ds_parser: GeneralDatasetParser):

        self.config     = config
        self.epochs     = config.num_epochs
        self.initial_lr = config.lr

        self.ds_parser  = ds_parser
        self.device     = config.device

        self.model      = hydra.utils.instantiate(config.model)
        self.model      = self.model.to(self.device)


        if self.config.training_mode == 'distillation':
            self.init_distillation_components()

        macs, params = get_model_complexity_info(self.model, (self.config.n_channels, 256, 256),\
                                                 as_strings = True, print_per_layer_stat = False, verbose = False)
        
        logging.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        logging.info('{:<30}  {:<8}'.format('Number of parameters: ',     params))

        self.loss_fn    = hydra.utils.instantiate(config.loss.seg_loss)
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr = self.initial_lr, weight_decay = 1e-5)

        self.train_dl    = self.get_dataloader('train')
        self.val_dl      = self.get_dataloader('val')

        self.eval_metric = Dice(num_classes = self.config.n_classes, average = 'macro').cuda()
        self.writer      = SummaryWriter(f'{self.config.log_location}/tensorboard')

        self.best_epoch  = 0
        self.best_dice   = 0.
        os.mkdir(f'{self.config.log_location}/model')
        

    #TODO
    def train(self):
        train_time1 = time.time()
        for epoch in range(1, self.epochs + 1):
            curr_lr = self.initial_lr * (1.0-np.float32(epoch-1)/np.float32(self.epochs))**(0.9)
            time1 = time.time()
            loss = self.train_one_epoch(curr_lr)
            time2 = time.time()
            logging.info('Epoch %d/%d, lr: %f, loss: %f' % (epoch, self.epochs, curr_lr, loss))
            logging.info('Epoch %d training time :%f minutes' % (epoch, np.round((time2-time1)/60, 2)))
            if epoch<self.epochs//4:
                continue
            time1 = time.time()
            dice_mean = self.validate()
            time2 = time.time()
            logging.info('Epoch %d validation time : %f minutes' % (epoch, np.round((time2-time1)/60, 2)))
            logging.info('Epoch %d validation Dice : %f ' % (epoch, dice_mean))
            self.writer.add_scalar('val/dice_mean', dice_mean,epoch)

            if dice_mean>= self.best_dice:
                self.best_epoch = epoch
                self.best_dice  = dice_mean
                torch.save(self.model.state_dict(), f'{self.config.log_location}/model/best_model.pth')

        logging.info('Best validation dice is: %f' % self.best_dice)
        logging.info('Best epoch is:           %d' % self.best_epoch)

        self.writer.close()
        train_time2 = time.time()
        training_time = (train_time2 - train_time1) / 3600
        logging.info('Training finished, tensorboardX writer closed')
        logging.info('Best epoch is %d, best mean dice is %f' % (self.best_epoch, self.best_dice))
        logging.info('Training total time: %f hours.' % training_time)

    def train_one_epoch(self,lr: float):

        self.model.train()
        for parm in self.optimizer.param_groups:
            parm['lr'] = lr
        loss_values = []
        for idx, batch in enumerate(self.train_dl):
            x, y,*x_missing = [item.float().cuda() for item in batch]

            if self.config.training_mode == 'distillation':
                x_missing = x_missing[0]
                features, logits = self.model(x_missing)
                with torch.no_grad():
                    features_t, logits_t = self.teacher_model(x)
                dice_loss, ce_loss, seg_loss     = self.loss_fn(logits, y)
                kd_loss                          = self.kd_loss(logits,logits_t)
                sim_map_s, sim_map_t, proto_loss = self.prototype_loss(features,features_t,y)
                loss = self.config.weights.seg * seg_loss + self.config.weights.kd * kd_loss + self.config.weights.proto * proto_loss
            else:
                features, logits = self.model(x)
                dice_loss, ce_loss, loss = self.loss_fn(logits, y)
            loss_values.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        epoch_loss = sum(loss_values)/len(loss_values)
        return epoch_loss            
        

    #TODO
    def validate(self):
        self.model.eval()
        self.eval_metric.reset()
        with torch.no_grad():
            for idx, batch in enumerate(self.val_dl):
                x, y, *x_missing = [item.float().cuda() for item in batch]
                _, logits = self.model(x_missing[0]) if self.config.training_mode == 'distillation' else self.model(x)
                y_pred    = logits.argmax(dim = 1)

                self.eval_metric.update(y_pred, y.squeeze().int())
            dice_mean = self.eval_metric.compute().item()
        return dice_mean



    def test(self):
        self.test_dl = self.get_dataloader('test')
        self.model.load_state_dict(torch.load(f'{self.config.log_location}/model/best_model.pth'))
        logging.info(f'Loading Best Model from epoch {self.best_epoch}')
        self.eval_metric.reset()
        with torch.no_grad():
            for idx, batch in enumerate(self.test_dl):
                x, y, *x_missing = [item.float().cuda() for item in batch]
                _, logits = self.model(x_missing[0]) if self.config.training_mode == 'distillation' else self.model(x)
                y_pred    = logits.argmax(dim = 1)

                self.eval_metric.update(y_pred, y.squeeze().int())
            self.test_dice = self.eval_metric.compute().item()
        logging.info('Dice score on the test set is: %f'%self.test_dice)

         

    def get_dataloader(self,mode: str):
        dataset = self.ds_parser.get_dataset(mode,self.config.training_mode)
        self.norm_params = dataset.norm_params

        if mode == "train":

            dataloader = data_utils.DataLoader(dataset,
                                               batch_size = self.config.batch_size,
                                               shuffle    = True,
                                               pin_memory = True,
                                               drop_last  = False
                                               )
        else:
            dataloader = data_utils.DataLoader(dataset,
                                               batch_size = self.config.batch_size,
                                               shuffle    = False,
                                               pin_memory = True,
                                               drop_last  = False
                                               )
        return dataloader


    def init_distillation_components(self):
        self.config.model.n_channels = self.config.n_teacher_channels
        self.teacher_model = hydra.utils.instantiate(self.config.model)
        self.teacher_model.load_state_dict(torch.load(self.config.teacher_model_path))
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        self.kd_loss    = hydra.utils.instantiate(self.config.loss.kd_loss) 
        self.prototype_loss = hydra.utils.instantiate(self.config.loss.prototype_loss)
