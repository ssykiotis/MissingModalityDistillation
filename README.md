# MissingModalityDistillation

Completed:

TODO:
Dataset Parser - done \
MMDDataset     - done \
Trainer        - in progress \
Loss functions - in progress \
Model          - done 

Random seed issue:
x,y are the same between 2 experiments.
Model weights initialization is the same between 2 experiments.
RESULTS ARE NOT. 
Source of the problem: Cross entropy loss not deterministic implementation
TODO: WRITE MY OWN CROSS ENTROPY IMPLEMENTATION. OR add relaxation constraint in save best model

TODO: Replace adaptive lr calculation with StepLR

TODO: FIX TENSORBOARD ENTRIES

TODO:
1) Either write my own cross entropy implementation or add relaxation constraint in save_best_model. Or: reduction = none and recude later. 
REDUCTION MANUALLY VS REDUCTION FROM F.cross_entropy GIVE TOTALLY DIFFERENT VALUES \ 
2) Models: model + teacher model if mode == distillation \
3) Homogenize loss functions. prepei na vrethei enas tropos na sou gyrnaei sygekrimeno arithmo outputs. \
4) update forward pass in train_one_epoch. Validation and test stay the same \
6) train_distillation_config: add teacher model path
7) Replace adaptive lr calculation with StepLR. Write my own class where step does both lr update and optimizer step. 



pretrain:       DiceCELoss (num_cls)
train_baseline: DiceCELoss (num_cls) 
train_protokd:  DiceCELoss(num_cls) dice + crossentropy +seg
                softmax_kl_loss
                prototype_loss