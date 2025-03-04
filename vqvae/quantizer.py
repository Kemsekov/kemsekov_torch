# taken from https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SonnetExponentialMovingAverage(nn.Module):
    # See: https://github.com/deepmind/sonnet/blob/5cbfdc356962d9b6198d5b63f0826a80acfdf35b/sonnet/src/moving_averages.py#L25.
    # They do *not* use the exponential moving average updates described in Appendix A.1
    # of "Neural Discrete Representation Learning".
    def __init__(self, decay, shape):
        super().__init__()
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros(*shape))
        self.register_buffer("average", torch.zeros(*shape))

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
        self.batch_norm = nn.BatchNorm1d(embedding_dim)
    
    def init_tensor(self,t):
        with torch.no_grad():
            t_n = F.normalize(torch.rand(t.shape).to(t.device),dim=0,p=2.0)*self.embedding_scale
            # t_n = torch.rand(t.shape).to(t.device)*self.embedding_scale
            t+=t_n-t
    
    def forward(self, x):
        # x is of shape (batch,emb_dim,height,width)

        batch_axis = 0
        emb_axis = 1
        dimensions_axis = [2,3,4][:len(x.shape)-2]
        
        flat_x = x.permute([batch_axis] + dimensions_axis + [emb_axis]).reshape(-1, self.embedding_dim)

        distances = (
            (flat_x ** 2).sum(1, keepdim=True)
            - 2 * flat_x @ self.e_i_ts
            + (self.e_i_ts ** 2).sum(0, keepdim=True)
        )
        encoding_indices = distances.argmin(1)
        ind = encoding_indices.view([x.shape[0]] + list(x.shape[2:]))
        quantized_x = F.embedding(
            ind, self.e_i_ts.transpose(0, 1)
        ).permute([0, len(x.shape)-1]+[1, 2, 3][:len(dimensions_axis)])

        quantized_x_d = x + (quantized_x - x).detach()
        
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
        