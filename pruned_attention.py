import torch
from torch import nn
import torch
import torch.nn.functional as F
from torch import nn
from einops.layers.torch import Rearrange
from kemsekov_torch.common_modules import ChanLayerNorm3D
from kemsekov_torch.residual import ResidualBlock
class PrunedCrossAttentionBlock(torch.nn.Module):
    def __init__(self,dim,mlp_dim,heads=8,dimensions=2,dropout=0.1,top_k=-1,normalization='batch'):
        """
        Somewhat optimal pruned cross-attention block
        
        dim: input dimensions
        
        mlp_dim: internal mlp dimension
        
        heads: heads for attention
        
        dimensions: dimensions count
        
        dropout: dropout to apply to attention layer
        
        top_k: count of elements to compute per dimension for each token
        """
        super().__init__()
        self.dpca = PrunedMultiheadAttention(dim,heads,dropout=dropout,top_k=top_k)
        self.mlp = torch.nn.Sequential(
            ResidualBlock(
                dim,
                [mlp_dim,dim],
                dimensions=dimensions,
                normalization=normalization,
                kernel_size=1
            ),
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
        attn = self.dpca(query_source,context)
        return self.mlp(attn)

class PrunedSelfAttentionBlock(torch.nn.Module):
    def __init__(self,dim,mlp_dim,heads=8,dimensions=2,dropout=0.1,top_k=-1,normalization='batch'):
        """
        Somewhat optimal pruned self-attention block
        
        dim: input dimensions
        
        mlp_dim: internal mlp dimension
        
        heads: heads for attention
        
        dimensions: dimensions count
        
        dropout: dropout to apply to attention layer
        
        top_k: count of elements to compute per dimension for each token
        """
        super().__init__()
        self.dpsa = PrunedSelfAttention(dim,dim//heads,heads,dropout=dropout,top_k=top_k)
        self.mlp = torch.nn.Sequential(
            ResidualBlock(
                dim,
                [mlp_dim,dim],
                dimensions=dimensions,
                normalization=normalization,
                kernel_size=1
            ),
        )
    def forward(self,x):
        """
        Computes multihead self-attention for given input x.
        """
        attn = self.dpsa(x)
        return self.mlp(attn)
    
def dist_to_random_Q_selection(Q, K, V, top_k : int):
    """
    Performs selection of keys and values based on distance of keys to random subset of queries.\n
    
    Q of shape [batch,length_q,dim] \n
    K,V of shape [batch,length_kv,dim]
    """
    if top_k>=K.shape[1]:
        return K,V
    B, tokens_count, DIM = Q.shape
    # Generate random indices for Q
    rand_token_ind = torch.randint(0, tokens_count, (B, min(top_k,tokens_count)), device=Q.device)
    
    # Select Q_small using advanced indexing
    Q_small = Q[torch.arange(B, device=Q.device)[:, None], rand_token_ind, :]  # [B, top_k, DIM]
    
    # Compute L1 distances using torch.cdist
    distances = torch.cdist(K, Q_small, p=1.0)  # [B, some_other_count, top_k]
    
    # Compute minimum distances over Q_small
    min_distances = distances.min(dim=2)[0]  # [B, some_other_count]
    
    # Get indices of top_k smallest minimum distances
    _, indices = torch.topk(min_distances, top_k, dim=1, largest=False, sorted=True)  # [B, top_k]
    
    all_batch_ind = torch.arange(B, device=K.device)[:, None]
    # Select corresponding points from K using advanced indexing
    selected_K = K[all_batch_ind, indices, :]  # [B, top_k, DIM]
    selected_V = V[all_batch_ind, indices, :]  # [B, top_k, DIM]
    
    return selected_K, selected_V
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
        top_k = 256,
        dropout: float = 0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
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
            add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
            add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
                Default: ``False``.
            kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
            vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
            batch_first: If ``True``, then the input and output tensors are provided
                as (batch, seq, feature). Default: ``True`` (batch, seq, feature).
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
        
        self.top_k=top_k
        
    def forward(self,query,keys,values):
        query_shape=query.shape

        query  = query.flatten(2).transpose(-1,-2)
        keys   = keys.flatten(2).transpose(-1,-2)
        values = values.flatten(2).transpose(-1,-2)
        
        keys,values = dist_to_random_Q_selection(query,keys,values,self.top_k)
        
        out,attn = self.model(query,keys,values)
        out = out.transpose(-1,-2).view(query_shape)
        
        return out