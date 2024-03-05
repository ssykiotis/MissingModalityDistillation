import logging
import time
import numpy as np
import hydra

from omegaconf import DictConfig
from src.dataset.DatasetParser import *
import torch.utils.data as data_utils
import logging

class Trainer:

    def __init__(self,config:DictConfig, ds_parser: GeneralDatasetParser):

        self.model = hydra.utils.instantiate(config.model)


        self.epochs     = config.epochs
        self.initial_lr = config.lr
        self.loss_fn    = hydra.utils.instantiate(config.loss)
        self.optimizer  = None

        self.train_dl = self.get_dataloader('train')
        self.val_dl   = self.get_dataloader('val')
        self.eval_metric = 
        
        pass

    #TODO
    def train(self):
        for epoch in range(1, self.epochs + 1):
            curr_lr = self.intial_lr * (1.0-np.float32(epoch)/np.float32(self.epochs))**(0.9)
            time1 = time.time()
            loss = self.train_one_epoch(curr_lr)
            logging.info('Epoch %d/%d, loss: %f' % (epoch, self.epochs, loss))
            time2 = time.time()
            logging.info('Epoch %d training time :%f minutes' % (epoch, np.round((time2-time1)/60),2))
            if epoch<self.epochs//4:
                continue
            self.validate()
        pass
        # time1 = time.time()
        # dice_evaluator = Dice(num_classes= num_cls, average = 'macro').cuda()
        # with torch.no_grad():
        #     for idx, sampled_batch in enumerate(val_loader):
        #         image,label = sampled_batch
        #         image,label = image.float().cuda(), label.float().cuda()
                        
        #         # predict,_ = test_single_case(model,image,STRIDE,CROP_SIZE,num_cls)
        #         _,predict = model(image)
        #         predict = predict.argmax(dim = 1)
        #         dice_evaluator.update(predict,label.squeeze().int())

        #         dice_mean = dice_evaluator.compute().item()


        #         # dice_wt,dice_co,dice_ec,dice_mean = eval_one_dice(predict,label)
        #         # dice_all_wt.append(dice_wt)
        #         # dice_all_co.append(dice_co)
        #         # dice_all_ec.append(dice_ec)
        #         # dice_all_mean.append(dice_mean)
        #         # logging.info('Sample [%d], average dice : %f' % (idx, dice_mean))
        # time2 = time.time()
        # logging.info('Epoch %d validation time : %f minutes' % (epoch, (time2-time1)/60))
        # logging.info('Epoch %d validation Dice : %f ' % (epoch, dice_mean))
    


    #         writer.add_scalar('val/dice_mean', dice_mean,epoch)
    #     if dice_mean>=best_dice:
    #         best_epoch = epoch
    #         best_dice = dice_mean
    #         # best_wt = dice_all_wt
    #         # best_co = dice_all_co
    #         # best_ec = dice_all_ec
    #         torch.save(model.state_dict(), save_model_path+'/best_model.pth')
    #     model.train()
    #     logging.info('Best dice is: %f'%best_dice)
    #     logging.info('Best epoch is: %d'%best_epoch)
    # writer.close()
    # train_time2 = time.time()
    # training_time = (train_time2 - train_time1) / 3600
    # logging.info('Training finished, tensorboardX writer closed')
    # logging.info('Best epoch is %d, best mean dice is %f'%(best_epoch,best_dice))
    # logging.info('Dice of wt/co/ec is %f,%f,%f'%(best_wt,best_co,best_ec))
    # logging.info('Training total time: %f hours.' % training_time)




    #TODO
    def train_one_epoch(self,lr: float):
        self.model.train()
        for parm in self.optimizer.param_groups:
            parm['lr'] = lr
        loss_values = []
        time1 = time.time()
        for idx, batch in enumerate(self.train_dl):
            x, y = [item.float().cuda() for item in batch]

            features, logits = self.model(x)
            dice_loss, ce_loss, loss = self.loss_fn(logits, y)
            loss_values.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        epoch_loss = sum(loss_values)/len(loss_values)
        return epoch_loss            
        

    #TODO
    def validate(self):
        pass

    #TODO
    def test(self):
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
