# MissingModalityDistillation

Completed:

TODO:
Dataset Parser - done \
MMDDataset     - done \
Trainer        - in progress \
Loss functions - in progress \
Model          - done 


TODO: Replace adaptive lr calculation with StepLR

TODO: FIX TENSORBOARD ENTRIES

TODO:



3) Homogenize loss functions. prepei na vrethei enas tropos na sou gyrnaei sygekrimeno arithmo outputs. \
6) train_distillation_config: add teacher model path
7) Replace adaptive lr calculation with StepLR. Write my own class where step does both lr update and optimizer step. 



pretrain:       DiceCELoss (num_cls)
train_baseline: DiceCELoss (num_cls) 
train_protokd:  DiceCELoss(num_cls) dice + crossentropy +seg
                softmax_kl_loss
                prototype_loss


replace .cuda() with .to(device)
prototypeloss n_classes from main config