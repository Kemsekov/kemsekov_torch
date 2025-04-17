import torch
from torch import nn

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
    def __init__(self,embed_dim,num_heads,top_k = 256):
        super().__init__()
        self.model=torch.nn.MultiheadAttention(embed_dim,num_heads,batch_first=True)
        self.top_k=top_k
        
    def forward(self,query,keys,values):
        query_shape=query.shape
        # x of shape [BATCH,CH,...dims...]
        query  = query.flatten(2).transpose(-1,-2)
        keys   = keys.flatten(2).transpose(-1,-2)
        values = values.flatten(2).transpose(-1,-2)
        
        keys,values = dist_to_random_Q_selection(query,keys,values,self.top_k)
        
        out,attn = self.model(query,keys,values)
        out = out.transpose(-1,-2).view(query_shape)
        
        return out