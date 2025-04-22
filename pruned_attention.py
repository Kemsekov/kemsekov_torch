import time
from typing import Tuple
import einops
import torch
from torch import nn
import torch
import torch.nn.functional as F
from torch import nn
from einops.layers.torch import Rearrange
from kemsekov_torch.residual import ResidualBlock

class PrunedSelfAttentionBlock(torch.nn.Module):
    def __init__(self,input_dim,mlp_dim,top_k=None,heads=8,dropout=0.1,device=None):
        """
        Somewhat optimal pruned self-attention block
        
        input_dim: input dimensions
        
        mlp_dim: internal mlp dimension
        
        top_k: count of elements to use for attention per head for each token. when set to `None` will use (input length) // heads
        
        heads: heads for attention
        
        qkdim: query/key dimension, when `None` defaults to `input_dim`
        
        dimensions: dimensions count
        
        dropout: dropout to apply to attention layer
        
        device: where to locate module
        """
        super().__init__()
        self.attn = PrunedCrossAttentionBlock(input_dim,mlp_dim,top_k,heads,dropout,device)
    def forward(self,x):
        return self.attn(x,x)
class PrunedCrossAttentionBlock(torch.nn.Module):
    def __init__(self,input_dim,mlp_dim,top_k=None,heads=8,dropout=0.1,device=None):
        """
        Somewhat optimal pruned cross-attention block
        
        input_dim: input dimensions
        
        mlp_dim: internal mlp dimension
        
        top_k: count of elements to use for attention per head for each token. when set to `None` will use (input length) // heads
        
        heads: heads for attention
        
        qkdim: query/key dimension, when `None` defaults to `input_dim`
        
        dimensions: dimensions count
        
        dropout: dropout to apply to attention layer
        
        device: where to locate module
        """
        super().__init__()
        self.Q = nn.Conv1d(
            input_dim,
            input_dim,
            kernel_size=1,
            device=device,
        )
        
        self.K = nn.Conv1d(
            input_dim,
            input_dim,
            kernel_size=1,
            device=device,
        )
        
        self.V = nn.Conv1d(
            input_dim,
            input_dim,
            kernel_size=1,
            device=device,
        )
        
        self.attn = PrunedMultiheadAttention(
            embed_dim=input_dim,
            num_heads=heads,
            dropout=dropout,
            top_k=top_k,
            device=device
        )
        self.attn_norm = torch.nn.BatchNorm1d(input_dim)
        
        self.attn_out_gamma = torch.nn.Parameter(torch.tensor(0.0,device=device))

        self.mlp = ResidualBlock(
            input_dim,
            [mlp_dim,input_dim],
            dimensions=1,
            kernel_size=1,
            dropout=dropout,
            normalization='batch', # batch,layer works well
            device=device,
        )
        
    def forward(self,query_source, context):
        """
        Computes multihead cross attention for given context and query source.
        
        query_source: tensor that is used to compute query(Q) of attention. 
        We need to embed information from context in this tensor. Output shape will be equal to query_source shape.
        
        context: tensor that is used to compute keys(K) and values(V) of attention. It is additional information that we need to embed into query_source.
        
        query_source shape can be != context shape, only batch and channel dimensions needs to match.
        
        When context==query_source, the results will be same as self-attention.
        """
        query_source_flat = query_source.flatten(2)
        Q = self.Q(query_source_flat)
        K = self.K(context.flatten(2))
        V = self.V(context.flatten(2))
        
        #--------------------
        # start = time.time()
        
        attn = self.attn(Q,K,V)
        attn=self.attn_norm(attn)
        
        attn = attn*self.attn_out_gamma+query_source_flat
        
        #--------------------
        # print("attn",time.time()-start)
        # start = time.time()
        
        result = self.mlp(attn)
        
        #--------------------
        # print("mlp",time.time()-start)
        
        return result.view(query_source.shape).contiguous()

# apparently best key,value pruning method
def sum_abs_prune(Q, K, V, top_k : int):
    q_probe = torch.sum(Q, dim=1)  # Reduce from (b, L, C) to (b, C) by summing over L
    k_abs = torch.abs(K) + K  # Element-wise absolute value plus original K, shape (b, L, C)
    score_l = torch.sum(q_probe[:, None, :] * k_abs, dim=2)  # Compute scores, shape (b, L)
    top_l_indices = score_l.topk(k=top_k, dim=-1).indices  # Get indices of top-k scores, shape (b, top_k)
    top_l_indices_expanded = top_l_indices[:, :, None].expand(-1, -1, K.shape[-1])  # Expand to (b, top_k, C)
    k_selected = torch.gather(K, dim=1, index=top_l_indices_expanded)  # Gather from K, shape (b, top_k, C)
    v_selected = torch.gather(V, dim=1, index=top_l_indices_expanded)  # Gather from V, shape (b, top_k, C)
    return k_selected, v_selected

class PrunedMultiheadAttention(nn.Module):
    """
    Pruned multihead attention.
    
    Accepts query, key and value as multidimensional tensor with channels.
    
    [BATCH,CHANNELS,DIM1,DIM2,...]
    """
    def __init__(
        self,
        embed_dim : int,
        num_heads : int,
        top_k = None,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = True,
        add_zero_attn: bool = True,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = True,
        device: int | None = None,
        dtype: int | None = None):
        """
        Args:
            embed_dim: Total dimension of the model.
            num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
                across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
            top_k: top K elements selection for keys
            dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
            bias: If specified, adds bias to input / output projection layers. Default: ``True``.
            add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``True``.
            add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
                Default: ``True``.
            kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
            vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
            batch_first: If ``True``, then the input and output tensors are provided
                as (batch, seq, feature). Default: ``True`` (batch, seq, feature).
        Accepts query, key and value as multidimensional tensor with channels.
        
        [BATCH,CHANNELS,DIM1,DIM2,...]
        """
        super().__init__()
        
        self.model=torch.nn.MultiheadAttention(
            embed_dim       =embed_dim,
            num_heads       =num_heads,
            dropout         =dropout,
            bias            =bias,
            add_bias_kv     =add_bias_kv,
            add_zero_attn   =add_zero_attn,
            kdim            =kdim,
            vdim            =vdim,
            batch_first     =batch_first,
            device          =device,
            dtype           =dtype,
        )
        self.num_heads=num_heads
        self.top_k = top_k
        
        #-------------------
        # self.extract_heads = Rearrange("B C ... -> B (...) C")
        
        self.extract_heads = Rearrange("B (h C) ... -> (B h) (...) C",h=num_heads)
        self.collect_heads = Rearrange("(B h) L C -> B L (h C)",h=num_heads)
        
    def forward(self,query,keys,values):
        query_shape=query.shape
        
        #--------------------
        # start = time.time()
        
        # extract dimension slices into separate head as a batch
        query  = self.extract_heads(query)
        keys   = self.extract_heads(keys)
        values = self.extract_heads(values)
        
        keys = F.normalize(keys, p=2.0, dim=-1) # [B, length_kv, DIM]
        query = F.normalize(query, p=2.0, dim=-1)  # [B, top_k, DIM]
        
        #--------------------
        # print("extract heads",time.time()-start)
        # start = time.time()
        
        top_k = query.shape[1]//self.num_heads if self.top_k is None else self.top_k
        # for each head to tokens selection
        # keys,values = dist_to_random_Q_selection_cosine(query, keys, values, top_k, top_k)
        keys,values = sum_abs_prune(query, keys, values, top_k)
    
        #--------------------
        # print("select tokens",time.time()-start)
        # start = time.time()
        
        # concat heads into full dimension
        query  = self.collect_heads(query)
        keys   = self.collect_heads(keys)
        values = self.collect_heads(values)
        
        #--------------------
        # print("concat heads",time.time()-start)
        # start = time.time()
        
        out,attn = self.model.forward(query,keys,values)
        out = out.transpose(-1,-2).view(query_shape)
        
        # out = query.transpose(-1,-2).view(query_shape)
        
        #--------------------
        # print("attention",time.time()-start)
        
        return out
