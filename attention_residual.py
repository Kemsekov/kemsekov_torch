import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
from kemsekov_torch.positional_emb import PositionalEncoding, AddPositionalEmbedding
class AttentionResidual(nn.Module):
    def __init__(self, dim, reduction_dim: int, residuals: Union[nn.Sequential, nn.ModuleList, List[nn.Module]]) -> None:
        """
        dim: data dimensions
        reduction_dim: data reduction dimension. If you passing images in shape [B,C,W,H] set it to 1 (channels), if \
            you processing timeseries of shape [B,seqlength,dim], set it to -1 (dim).\
            It must be a 'representative'\
            dimension that all other dimensions (except for batch) can be reduced to by mean.
        residuals: list of modules to apply attention residual
        """
        super().__init__()
        # use learned queries for each residual, initialized to abs positional encoding
        self.queries = nn.Sequential(
            nn.Linear(dim, dim),
            nn.RMSNorm(dim)
        )
        self.query_pos = nn.Parameter(torch.randn((len(residuals),1,dim)))

        # each residual intermediate value is reduced and key for it computed via
        # this linear mapping
        self.key = nn.Sequential(
            nn.Linear(dim, dim),
            nn.RMSNorm(dim)
        )
        if isinstance(residuals,list):
            residuals=nn.ModuleList(residuals)
            
        self.residuals = residuals
        self.reduction_dim = reduction_dim

    def forward(self, x):
        input_shape = x.shape
        B = x.shape[0]

        if x.ndim > 2:
            mean_dims = list([i for i in range(1, x.ndim) if i != self.reduction_dim % x.ndim])
            x_mean = x.mean(mean_dims)
        else:
            x_mean = x
            mean_dims = None

        x_flat = x.view(B, -1)
        x_key = self.key(x_mean)

        # pre-allocate torch buffers
        keys = torch.empty((B, len(self.residuals) + 1, x_key.shape[-1]), device=x.device, dtype=x.dtype)
        values = torch.empty((B, len(self.residuals) + 1, x_flat.shape[-1]), device=x.device, dtype=x.dtype)

        # init input x
        keys[:, 0] = x_key
        values[:, 0] = x_flat
        
        
        # Reset version counter so subsequent inplace writes don't trigger
        # autograd's version mismatch error. The values slice views see the
        # correct data (writes are always to indices outside the slice range),
        # so resetting the version is safe — it just tells autograd "nothing
        # changed" which is true for the slice's data range.
        torch._C._autograd._unsafe_set_version_counter([keys], [0])
        torch._C._autograd._unsafe_set_version_counter([values], [0])

        for ind, layer in enumerate(self.residuals):
            query=x_mean+self.query_pos[ind]
            # [batch,1,dim]
            query = query[:,None]
            # generate query from previous module output + current module position
            
            # [batch,seqlen,dim] — view, no clone
            ks = keys[:, :ind + 1]
            # [batch,seqlen,val_dim] — view, no clone
            vals = values[:, :ind + 1]

            # reshape query keys and values to 1-head inputs for attention
            query = query[:, None]
            ks = ks[:, None]
            vals = vals[:, None]

            # compute cross-attention
            out = F.scaled_dot_product_attention(query, ks, vals).reshape(input_shape)
            x = layer(out)

            # update keys and values history with newly generated data
            x_mean = x if mean_dims is None else x.mean(mean_dims)

            keys[:, ind + 1] = self.key(x_mean)
            values[:, ind + 1] = x.view(B, -1)

            # Reset version counter after each inplace write to values buffer.
            # The write is to index ind+1, which is outside all previously-saved
            # slice ranges (0..ind). So the saved slices' data is unchanged —
            # resetting the version just prevents the false-positive mismatch.
            torch._C._autograd._unsafe_set_version_counter([keys], [0])
            torch._C._autograd._unsafe_set_version_counter([values], [0])
            
        return x

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

