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


pretrain:       DiceCELoss (num_cls)
train_baseline: DiceCELoss (num_cls) 
train_protokd: DiceCELoss(num_cls) dice + crossentropy +seg
                softmax_kl_loss
                prototype_loss