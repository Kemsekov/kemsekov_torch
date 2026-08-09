import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
from typing import Iterable
from kemsekov_torch.common_modules import Residual

class AttentionResidual(nn.Module):
    def __init__(
        self, 
        modules : Iterable[nn.Module],
        features_dim,
        features_dimension=1,
    ):
        """
        for (B,C,H,W) images use features_dim=C,features_dimension=1
        
        for (B,L,C) where L is sequence length, use features_dim=C, features_dimension=2
        
        for (B,C) vectors use features_dim=C, features_dimension=1
        """
        super().__init__()
        self.models = nn.ModuleList(modules)
        self.query = nn.Parameter(torch.randn(len(modules)+1,features_dim))
        # nn.init.orthogonal_(self.query)
        self.KV = nn.Sequential(
            nn.RMSNorm(features_dim),
            nn.SiLU(),
            nn.Linear(features_dim,features_dim,bias=False),
        )
        # self.key_norm = nn.RMSNorm(features_dim)
        
        self.out = nn.Sequential(
            nn.RMSNorm(features_dim),
            Residual([
                nn.SiLU(),
                nn.Linear(features_dim,features_dim)
            ])
        )
        
        self.features_dimension=features_dimension
        self.head_dim=features_dim
        
    def forward(self,x : torch.Tensor):
        #xt is [B,...,head_dim]
        xt=x.transpose(self.features_dimension,-1)
        
        keys = []
        values = []
        #key/values is of shape [|models|,1,(B,...),head_dim]
        
        for i,m in enumerate(self.models):
            k = self.KV(xt)
            v=xt
            q = self.query[i]
            #q,k,v of shape [(B,...),head_dim]
            k=F.normalize(k,2.0,-1)
            keys.append(k)
            values.append(v)
            
            if i>0:
                x_next = self.get_x_next(keys, values, q)
            else:
                x_next = self.out(v).transpose(-1,self.features_dimension)
                
            x = m(x_next)
            xt=x.transpose(self.features_dimension,-1)
        
        k = self.KV(xt)
        v=xt
        keys.append(k)
        values.append(v)
        # return xt
        return self.get_x_next(keys, values, self.query[-1])

    def get_x_next(self, keys, values, q:torch.Tensor):
        keys=torch.stack(keys)
        scores = (keys*q.unsqueeze(0)).mean(-1,keepdim=True)
        out = torch.stack(values,0)*scores.softmax(0)
        out = self.out(out.sum(0))
        return out.transpose(-1,self.features_dimension)