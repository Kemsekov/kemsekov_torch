import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
from kemsekov_torch.positional_emb import PositionalEncoding, AddPositionalEmbedding

class AttentionResidual(nn.Module):
    def __init__(self, dim, residuals: Union[nn.Sequential, nn.ModuleList, List[nn.Module]]) -> None:
        """
        dim: data dimensions
        residuals: list of modules to apply attention residual. Each module in `residuals` **must not return residual** like `f(x)+x`, else it\
            will kill gradients flow
        """
        super().__init__()
        init_scale = 0.3
        # use learned queries for each residual, initialized to abs positional encoding
        self.query_pos = nn.Parameter(torch.randn((len(residuals)+1,1,dim))*init_scale)
        self.key_pos = nn.Parameter(torch.randn((1,len(residuals)+1,dim))*init_scale)

        # each residual intermediate value is reduced and key for it computed via
        # this linear mapping
        if isinstance(residuals,list):
            residuals=nn.ModuleList(residuals)
            
        self.residuals = residuals

    def forward(self, x):
        input_shape = x.shape
        B = x.shape[0]

        x_flat = x.view(B, -1)

        values = torch.empty((B, len(self.residuals) + 1, x_flat.shape[-1]), device=x.device, dtype=x.dtype)

        values[:, 0] = x_flat
        
        torch._C._autograd._unsafe_set_version_counter([values], [0])

        for ind, layer in enumerate(self.residuals):
            query=self.query_pos[ind][[0]*B]
            # [batch,1,dim]
            query = query[:,None]
            
            # [batch,seqlen,dim] — view, no clone
            ks = self.key_pos[:,:ind + 1][[0]*B]
            # [batch,seqlen,val_dim] — view, no clone
            vals = values[:, :ind + 1]

            # reshape query keys and values to 1-head inputs for attention
            query = query[:, None]
            ks = ks[:, None]
            vals = vals[:, None]

            # compute cross-attention
            out = F.scaled_dot_product_attention(query, ks, vals).reshape(input_shape)
            x = layer(out)

            values[:, ind + 1] = x.view(B, -1)

            torch._C._autograd._unsafe_set_version_counter([values], [0])
        
        query=self.query_pos[-1][[0]*B]
        # [batch,1,dim]
        query = query[:,None]
        
        # [batch,seqlen,dim] — view, no clone
        ks = self.key_pos[[0]*B]
        # [batch,seqlen,val_dim] — view, no clone
        vals = values

        # reshape query keys and values to 1-head inputs for attention
        query = query[:, None]
        ks = ks[:, None]
        vals = vals[:, None]

        # compute cross-attention
        # print(query.shape,ks.shape,vals.shape)
        out = F.scaled_dot_product_attention(query, ks, vals).reshape(input_shape)
        torch._C._autograd._unsafe_set_version_counter([values], [0])
            
        return out

class GRUResidual(nn.Module):
    def __init__(self, dim,reduction_dim : int, residuals : Union[nn.Sequential,nn.ModuleList,List[nn.Module]]) -> None:
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
        if isinstance(residuals,list):
            residuals = nn.ModuleList(residuals)
        self.residuals = residuals
        self.reduction_dim=reduction_dim
        self.transaction=nn.Linear(2*dim,2)
    
    def forward(self,x):
        if x.ndim>2:
            mean_dims = list([i for i in range(1,x.ndim) if i!=self.reduction_dim%x.ndim])
        else:
            mean_dims = None
        
        x_state = x
        
        for layer in self.residuals:

            y_state = layer(x_state)

            x_key = self.to_key(x_state.mean(mean_dims) if mean_dims is not None else x_state)
            y_key = self.to_key(y_state.mean(mean_dims) if mean_dims is not None else y_state)
            
            transaction_pair = torch.concat([x_key,y_key],-1)
            transaction_value = self.transaction(transaction_pair).sigmoid()
            
            while transaction_value.ndim!=x.ndim:
                transaction_value=transaction_value.unsqueeze(-2)
            keep_old,keep_new = transaction_value.chunk(2,-1)
            
            x_state = x_state*keep_old+y_state*keep_new
            
        return x_state

