import torch
from   torch import nn
import torch.nn.functional as F

class PrototypeLoss(nn.Module):
    def __init__(self,n_classes: int, eps = 1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.eps       = eps
    
    def forward(self,f_input:torch.tensor, f_target:torch.tensor,label:torch.tensor)-> torch.tensor:

        N     = len(f_input.size()) - 2
        label = label[:,0]  #label [N,*]
        s     = []
        t     = []

        for i in range(self.n_classes):
            mask = (label == i)
            if N==2 and (torch.sum(mask,dim=(-2,-1))>0).all():
                proto_s =  torch.sum(f_input  * mask[:, None], dim=(-2,-1)) / (torch.sum(mask[:, None], dim = (-2,-1)) + self.eps)
                proto_t =  torch.sum(f_target * mask[:, None], dim=(-2,-1)) / (torch.sum(mask[:, None], dim = (-2,-1)) + self.eps)
                proto_map_s = F.cosine_similarity(f_input,   proto_s[:,:, None, None], dim = 1, eps = self.eps)
                proto_map_t = F.cosine_similarity(f_target , proto_t[:,:, None, None], dim = 1, eps = self.eps)
                s.append(proto_map_s.unsqueeze(1))
                t.append(proto_map_t.unsqueeze(1))
        sim_map_s = torch.cat(s, dim = 1)
        sim_map_t = torch.cat(t, dim = 1)
        loss = torch.mean( (sim_map_s - sim_map_t) ** 2)
        return sim_map_s, sim_map_t, loss