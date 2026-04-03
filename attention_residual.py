import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from kemsekov_torch.positional_emb import PositionalEncoding

class AttentionResidual(nn.Module):
    def __init__(self, dim,reduction_dim : int, residuals : Union[nn.Sequential,nn.ModuleList]) -> None:
        """
        dim: data dimensions
        reduction_dim: data reduction dimension. If you passing images in shape [B,C,W,H] set it to 1 (channels), if \
            you processing timeseries of shape [B,seqlength,dim], set it to -1 (dim).\
            It must be a 'representative'\
            dimension that all other dimensions (except for batch) can be reduced to by mean.
        residuals: list of modules to apply attention residual
        """
        super().__init__()
        pos = PositionalEncoding(dim,freq=len(residuals))(torch.zeros((1,len(residuals),dim)))[0]
        
        # use learned queries for each residual, initialized to abs positional encoding
        self.queries=nn.Parameter(pos)
        
        # each residual intermediate value is reduced and key for it computed via
        # this linear mapping
        self.key = nn.Linear(dim,dim)
        
        self.residuals = residuals
        self.reduction_dim=reduction_dim
    
    def forward(self,x):
        input_shape = x.shape
        B = x.shape[0]
        
        if x.ndim>2:
            mean_dims = list([i for i in range(1,x.ndim) if i!=self.reduction_dim%x.ndim])
            x_mean = x.mean(mean_dims)
        else:
            x_mean=x
            mean_dims = None
        
        x_flat = x.view(B,-1)
        x_key = self.key(x_mean)

        # pre-allocate torch buffers        
        keys = torch.empty((B,len(self.residuals)+1,x_key.shape[-1]),device=x.device,dtype=x.dtype)
        values = torch.empty((B,len(self.residuals)+1, x_flat.shape[-1]),device=x.device,dtype=x.dtype)
        
        #init input x
        keys[:,0]=x_key
        values[:,0]=x_flat
        
        for ind,(layer,query) in enumerate(zip(self.residuals,self.queries)):
            # [batch,1,dim]
            query = query[None,None,:][[0]*B]
            # [batch,seqlen,dim]
            ks = keys[:,:ind+1]
            # [batch,seqlen,val_dim]
            vals = values[:,:ind+1]

            #reshape query keys and values to 1-head inputs for attention
            #(BATCH_SIZE, HEADS_NUM, LENGTH, HEAD_DIM)
            query = query[:,None]
            ks = ks[:,None]
            vals = vals[:,None]
            
            # compute cross-attention with new token (query) and all previous
            # tokens representatives (ks), and by their relations combine
            # all previous values (vals) into a new token.
            out = F.scaled_dot_product_attention(query,ks,vals).reshape(input_shape)
            x = layer(out)
            
            # update keys and values history with newly generated data
            if mean_dims is not None:
                x_mean = x.mean(mean_dims)
            else:
                x_mean=x
            
            keys[:,ind+1]=(self.key(x_mean))
            values[:,ind+1]=x.view(B,-1)
            
        return x



class GRUResidual(nn.Module):
    def __init__(self, dim,reduction_dim : int, residuals : Union[nn.Sequential,nn.ModuleList]) -> None:
        """
        dim: data dimensions
        reduction_dim: data reduction dimension. If you passing images in shape [B,C,W,H] set it to 1 (channels), if \
            you processing timeseries of shape [B,seqlength,dim], set it to -1 (dim).\
            It must be a 'representative'\
            dimension that all other dimensions (except for batch) can be reduced to by mean.
        residuals: list of modules to apply gru residual
        """
        super().__init__()
        
        # each residual intermediate value is reduced and key for it computed via
        # this linear mapping
        self.to_key = nn.Linear(dim,dim)
        
        self.residuals = residuals
        self.reduction_dim=reduction_dim
        self.transaction=nn.Linear(2*dim,2)
    
    def forward(self,x):
        if x.ndim>2:
            mean_dims = list([i for i in range(1,x.ndim) if i!=self.reduction_dim%x.ndim])
            x_mean = x.mean(mean_dims)
        else:
            x_mean=x
            mean_dims = None
        
        x_state = x
        x_key = self.to_key(x_mean)
        
        for layer in self.residuals:
            y_state = layer(x_state)
            y_key = self.to_key(y_state.mean(mean_dims) if mean_dims is not None else y_state)
            transaction_pair = torch.concat([x_key,y_key],-1)
            transaction_value = self.transaction(transaction_pair).softmax(-1)
            while transaction_value.ndim!=x.ndim:
                transaction_value=transaction_value.unsqueeze(-2)
            keep_old,keep_new = transaction_value.chunk(2,-1)
            
            x_state = x_state*keep_old+y_state*keep_new
            
        return x_state
