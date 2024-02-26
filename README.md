# MissingModalityDistillation

Completed:

TODO:
Dataset Parser - done \
MMDDataset     - done \
Trainer        - in progress \
Loss functions - in progress \
Model          - done 


pretrain:       DiceLoss (num_cls)
train_baseline: DiceLoss (num_cls) (dic + crossentropy)
train_protokd: DiceCELoss(num_cls) dice + crossentropy +seg
                softmax_kl_loss
                prototype_loss