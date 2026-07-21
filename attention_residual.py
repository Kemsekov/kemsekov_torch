import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
from typing import Iterable
class AttentionResidual1(nn.Module):
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
        self.QKV = nn.Sequential(
            nn.RMSNorm(features_dim),
            nn.Linear(features_dim,features_dim*3,bias=False),
        )
        
        self.features_dimension=features_dimension
        self.head_dim=features_dim
        
    def forward(self,x : torch.Tensor):
        #xt is [B,...,head_dim]
        xt=x.transpose(self.features_dimension,-1)
        input_dimension_prod = x.numel()//self.head_dim
        
        keys = torch.zeros((len(self.models)+1,1,input_dimension_prod,self.head_dim),device=x.device,dtype=x.dtype)
        values = torch.zeros((len(self.models)+1,1,input_dimension_prod,self.head_dim),device=x.device,dtype=x.dtype)
        #key/values is of shape [|models|,1,(B,...),head_dim]
        
        # these ugly autograd things allows us to use efficient preallocated buffers for
        # keys and values with inplace operations, without having autograd problems
        torch._C._autograd._unsafe_set_version_counter([keys], [0])
        torch._C._autograd._unsafe_set_version_counter([values], [0])
        
        for i,m in enumerate(self.models):
            q,k,v = self.QKV(xt).view(-1,self.head_dim*3).chunk(3,-1)
            #q,k,v of shape [(B,...),head_dim]
            keys[i]=k.unsqueeze(0)
            values[i]=v.unsqueeze(0)
            torch._C._autograd._unsafe_set_version_counter([keys], [0])
            torch._C._autograd._unsafe_set_version_counter([values], [0])
            
            if i>0:
                x_next = self.get_x_next(xt, keys, values, i, q)
            else:
                x_next = x
                
            x = m(x_next)
            xt=x.transpose(self.features_dimension,-1)
        
        q,k,v = self.QKV(xt).view(-1,self.head_dim*3).chunk(3,-1)
        keys[-1]=k.unsqueeze(0)
        values[-1]=v.unsqueeze(0)
        torch._C._autograd._unsafe_set_version_counter([keys], [0])
        torch._C._autograd._unsafe_set_version_counter([values], [0])
        # return xt
        return self.get_x_next(xt, keys, values, len(self.models)+1, q)

    def get_x_next(self, xt, keys, values, i, q):
        prev_keys=keys[:i]
        prev_values=values[:i]
        #key/values is of shape [i,1,(B,...),,head_dim]
            
        prev_keys=prev_keys.transpose(0,2)
        prev_values=prev_values.transpose(0,2)
        #key/values is of shape [(B,...),1,i,head_dim]
            
        q = q.view(-1,1,1,self.head_dim)
        #q is of shape [(B,...),1,1,head_dim]
        x_next = nn.functional.scaled_dot_product_attention(q,prev_keys,prev_values)[:,0,0]
        x_next=x_next.view(xt.shape).transpose(-1,self.features_dimension)

        return x_next

