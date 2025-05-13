# taken from https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
import random
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
    # Compute norms
    e_norm = e.norm(p=2.0, dim=-1, keepdim=True).detach().clamp(1e-6)  # (8, 32, 32, 1)
    q_norm = q.norm(p=2.0, dim=-1, keepdim=True).detach().clamp(1e-6)  # (8, 32, 32, 1)
    
    # Unit vectors
    e_hat = (e / e_norm).detach()  # (8, 32, 32, 5)
    q_hat = (q / q_norm).detach()  # (8, 32, 32, 5)
    
    # Scaling factor
    lambda_val = q_norm / e_norm  # (8, 32, 32, 1)
    
    # Correct reflection vector
    e_hat_plus_q_hat = e_hat + q_hat  # (8, 32, 32, 5)
    norm_e_hat_plus_q_hat = e_hat_plus_q_hat.norm(p=2.0, dim=-1, keepdim=True).detach().clamp(1e-6)  # (8, 32, 32, 1)
    r = (e_hat_plus_q_hat / norm_e_hat_plus_q_hat).detach()  # (8, 32, 32, 5)
    
    # Compute r^T e directly as dot product
    r_dot_e = (r * e).sum(dim=-1, keepdim=True)  # (8, 32, 32, 1)
    
    # Expression terms (keep shapes consistent)
    r_r_e = r * r_dot_e  # (8, 32, 32, 5) * (8, 32, 32, 1) -> (8, 32, 32, 5)
    q_hat_e_norm = q_hat * e_norm  # (8, 32, 32, 5) * (8, 32, 32, 1) -> (8, 32, 32, 5)
    
    # Compute expression
    expression = e - 2 * r_r_e + 2 * q_hat_e_norm  # (8, 32, 32, 5)
    q_result = lambda_val * expression  # (8, 32, 32, 1) * (8, 32, 32, 5) -> (8, 32, 32, 5)
    
    return q_result

def skip_gradient_trick(e,q):
    return e+(q-e).detach()

def skip_plus_rotation_trick(e,q):
    return 0.5*(skip_gradient_trick(e,q)+rotation_trick(e,q))

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
        
        self.register_buffer("e_i_ts", e_i_ts)

        # Exponential moving average of the cluster counts.
        self.cluster_counts = SonnetExponentialMovingAverage(decay, (num_embeddings,))
        # Exponential moving average of the embeddings.
        self.embeddings = SonnetExponentialMovingAverage(decay, e_i_ts.shape)
    
    def forward(self, x):
        """
        x of shape [batch,emb_dim,...dims...], supports up to 5 spatial dimensions
        
        returns quantized_x, quantized_indices
        """
        # x is of shape (batch,emb_dim,height,width)

        batch_axis = 0
        emb_axis = 1
        dimensions_axis = [2,3,4,5,6][:len(x.shape)-2]
        
        x_permute = x.permute([batch_axis] + dimensions_axis + [emb_axis])
        permute_to_orig = [0, len(x.shape)-1]+[1, 2, 3, 4, 5][:len(dimensions_axis)]
        
        flat_x = x_permute.reshape(-1, self.embedding_dim)

        distances = (
            (flat_x ** 2).sum(1, keepdim=True)
            - 2 * flat_x @ self.e_i_ts
            + (self.e_i_ts ** 2).sum(0, keepdim=True)
        )
        encoding_indices = distances.argmin(1)
        #same
        # print(x_permute.shape)
        # print(quantized_x.shape)
        
        ind = encoding_indices.view([x.shape[0]] + list(x.shape[2:]))
        
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

        quantized_x = F.embedding(
            ind, self.e_i_ts.transpose(0, 1)
        )
        
        # quantized_x_d = skip_gradient_trick(x_permute,quantized_x)
        quantized_x_d = rotation_trick(x_permute,quantized_x)
        # quantized_x_d = skip_plus_rotation_trick(x_permute,quantized_x)
        
        quantized_x=quantized_x.permute(permute_to_orig)
        quantized_x_d=quantized_x_d.permute(permute_to_orig)
        
        # return detached quantized and just quantized
        # quantized_x_d can be passed to decoder,
        # quantized_x can be used to update codebook
        # ind is id of quantized vectors
        return quantized_x_d, ind
    
    @torch.jit.export
    def decode_from_ind(self,ind):
        """
        Decodes tensor from indices
        """
        dimensions_axis = [2,3,4,5,6][:len(ind.shape)-1]
        permute_to_orig = [0, len(ind.shape)]+[1, 2, 3, 4, 5][:len(dimensions_axis)]
        quantized_x = F.embedding(
            ind, self.e_i_ts.transpose(0, 1)
        )
        return quantized_x.permute(permute_to_orig)
    
    @torch.jit.export
    def get_codebook_usage(self,threshold : float=1e-5):
        """
        Returns count of used codebooks, 
        count of codebooks, that relative to most used codebook have usage > `threshold`
        """
        usage = self.cluster_counts.average.clone().detach()
        usage/=usage.max()
        used = torch.where(usage>threshold)[0].numel()
        return used