# this is just convinence file that imports all general nessesary classes to use linear attention
from kemsekov_torch.attention import LinearSelfAttentionBlock, LinearCrossAttentionBlock

# FlattenSpatialDimensions - allows to convert multidim tensors of shape [batch,channels,...]
# to input sequences of shape [batch,seq_len,channels], feed them to linear attn and reshape back
# where seq_len is flatten of (...)

# Permute allows to do dimensions permutation of input tensors
# Transpose allows to do transpose of tensor dimensions
from kemsekov_torch.common_modules import FlattenSpatialDimensions, Permute, Transpose

# LearnedPosEmb allows to add learned positional embedding to tensors of shape [batch,seq_len,channels]
from learned_emb import LearnedPosEmb
