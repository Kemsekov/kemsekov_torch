# taken from https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def rotation_trick(e, q):
    """
    Applies the rotation trick to transform e into q for VQVAE.
    
    Args:
        e (torch.Tensor): Input tensor of shape (batch, height, width, hidden), e.g., (8, 32, 32, hidden)
        q (torch.Tensor): Target tensor of shape (batch, height, width, hidden), e.g., (8, 32, 32, hidden)
    
    Returns:
        torch.Tensor: Transformed tensor q_result of shape (batch, height, width, hidden)
    """
    # Compute L2 norms along the hidden dimension, shape: (8, 32, 32, 1)
    e_norm = e.norm(p=2.0, dim=-1, keepdim=True).detach()+1e-6
    q_norm = q.norm(p=2.0, dim=-1, keepdim=True).detach()+1e-6
    
    # Compute unit vectors, shape: (8, 32, 32, hidden)
    e_hat = (e / e_norm).detach()
    q_hat = (q / q_norm).detach()
    
    # Compute scaling factor lambda, shape: (8, 32, 32, 1)
    lmbda = q_norm / e_norm
    
    # Compute reflection direction r, shape: (8, 32, 32, hidden)
    r = e_hat + q_hat
    r/=r.norm(p=2.0, dim=-1, keepdim=True)
    
    # Compute outer products, shape: (8, 32, 32, hidden, hidden)
    r_rT = r.unsqueeze(-1) * r.unsqueeze(-2)
    q_hat_e_hatT = q_hat.unsqueeze(-1) * e_hat.unsqueeze(-2)
    
    e_unsqueeze = e.unsqueeze(-1)
    # Apply rotation to e, shape: (8, 32, 32, hidden)
    q_result = lmbda*(e_unsqueeze-2*r_rT @ e_unsqueeze+2*q_hat_e_hatT @ e_unsqueeze).squeeze(-1)
    
    return q_result

class SonnetExponentialMovingAverage(nn.Module):
    # See: https://github.com/deepmind/sonnet/blob/5cbfdc356962d9b6198d5b63f0826a80acfdf35b/sonnet/src/moving_averages.py#L25.
    # They do *not* use the exponential moving average updates described in Appendix A.1
    # of "Neural Discrete Representation Learning".
    def __init__(self, decay, shape,init_hidden = None,init_average = None):
        super().__init__()
        self.decay = decay
        self.counter = 0
        if init_hidden is None:
            init_hidden = torch.zeros(*shape)
        if init_average is None:
            init_average = torch.zeros(*shape)
        self.register_buffer("hidden", init_hidden)
        self.register_buffer("average", init_average)

    def update(self, value):
        if self.training:
            self.counter += 1
            with torch.no_grad():
                self.hidden -= (self.hidden - value) * (1 - self.decay)
                self.average = self.hidden / (1 - self.decay ** self.counter)

    def __call__(self, value):
        self.update(value)
        return self.average
    def forward(self,x):
        self.update(x)
        return self.average
    
class VectorQuantizer(nn.Module):
    # embedding_scale is important factor that must match scales of encoder outputs embeddings
    # over embedding dimension
    def __init__(self, embedding_dim, num_embeddings, decay=0.99, epsilon=1e-5,embedding_scale = 1):
        super().__init__()
        # See Section 3 of "Neural Discrete Representation Learning" and:
        # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L142.

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        # Weight for the exponential moving average.
        self.decay = decay
        # Small constant to avoid numerical instability in embedding updates.
        self.epsilon = epsilon
        self.embedding_scale=embedding_scale

        # Dictionary embeddings.
        e_i_ts = torch.FloatTensor(embedding_dim, num_embeddings)
        self.init_tensor(e_i_ts)
        
        self.register_buffer("e_i_ts", e_i_ts)

        # Exponential moving average of the cluster counts.
        self.cluster_counts = SonnetExponentialMovingAverage(decay, (num_embeddings,))
        # Exponential moving average of the embeddings.
        self.embeddings = SonnetExponentialMovingAverage(decay, e_i_ts.shape)
    
    def init_tensor(self,t : torch.Tensor):
        with torch.no_grad():
            t_n = F.normalize(torch.rand(t.shape).to(t.device),dim=1,p=2.0)*self.embedding_scale
            # t_n = torch.rand(t.shape).to(t.device)*self.embedding_scale
            t.zero_()
            t+=t_n
    
    def forward(self, x):
        # x is of shape (batch,emb_dim,height,width)

        batch_axis = 0
        emb_axis = 1
        dimensions_axis = [2,3,4][:len(x.shape)-2]
        
        x_permute = x.permute([batch_axis] + dimensions_axis + [emb_axis])
        flat_x = x_permute.reshape(-1, self.embedding_dim)

        distances = (
            (flat_x ** 2).sum(1, keepdim=True)
            - 2 * flat_x @ self.e_i_ts
            + (self.e_i_ts ** 2).sum(0, keepdim=True)
        )
        permute_to_orig = [0, len(x.shape)-1]+[1, 2, 3][:len(dimensions_axis)]
        encoding_indices = distances.argmin(1)
        ind = encoding_indices.view([x.shape[0]] + list(x.shape[2:]))
        quantized_x = F.embedding(
            ind, self.e_i_ts.transpose(0, 1)
        )
        
        if self.training:
            with torch.no_grad():
                # See Appendix A.1 of "Neural Discrete Representation Learning".
                # Cluster counts.
                encoding_one_hots = F.one_hot(
                    encoding_indices, self.num_embeddings
                ).type(flat_x.dtype)
                n_i_ts = encoding_one_hots.sum(0)
                # Updated exponential moving average of the cluster counts.
                # See Equation (6).
                self.cluster_counts(n_i_ts)

                # Exponential moving average of the embeddings. See Equation (7).
                embed_sums = flat_x.transpose(0, 1) @ encoding_one_hots
                self.embeddings(embed_sums)

                # This is kind of weird.
                # Compare: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L270
                # and Equation (8).
                N_i_ts_sum = self.cluster_counts.average.sum()
                N_i_ts_stable = (
                    (self.cluster_counts.average + self.epsilon)
                    / (N_i_ts_sum + self.num_embeddings * self.epsilon)
                    * N_i_ts_sum
                )
                self.e_i_ts = self.embeddings.average / N_i_ts_stable.unsqueeze(0)
        #same
        # print(x_permute.shape)
        # print(quantized_x.shape)
        quantized_x_d=rotation_trick(x_permute,quantized_x)
        quantized_x=quantized_x.permute(permute_to_orig)
        quantized_x_d=quantized_x_d.permute(permute_to_orig)
        
        # return detached quantized and just quantized
        # quantized_x_d can be passed to decoder,
        # quantized_x can be used to update codebook
        # ind is id of quantized vectors
        return quantized_x_d,quantized_x, ind
    
    def get_codebook_usage(self,threshold=1e-5):
        """
        Returns count of used codebooks, 
        count of codebooks, that relative to most used codebook have usage > `threshold`
        """
        usage = self.cluster_counts.average.clone().detach()
        usage/=usage.max()
        used = torch.where(usage>threshold)[0].numel()
        return used
    
    def update_unused_codebooks(self, amount=0.05, threshold=1e-5):
        """
        Updates codebooks with usage smaller than threshold.
        threshold equals to a fraction of a max usage, so threshold=0.01 means
        tokens that is used 100 times less frequent than than most used token
        """
        # Determine the maximum number of embeddings to reinitialize.
        num_to_reinit = int(self.num_embeddings * amount)
        counts = self.cluster_counts.average.detach().clone()
        counts/=counts.max()
        
        unused_indices = torch.where(counts < threshold)[0][:num_to_reinit]
        
        with torch.no_grad():
            self.init_tensor(self.e_i_ts[:,unused_indices])
            
            self.cluster_counts.average[unused_indices]=0
            self.cluster_counts.hidden[unused_indices]=0

            self.embeddings.hidden[:,unused_indices]=0
            self.embeddings.average[:,unused_indices]=0
        