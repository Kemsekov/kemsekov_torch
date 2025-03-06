from kemsekov_torch.vqvae.quantizer import VectorQuantizer
import torch.nn as nn
class VQVAE2(nn.Module):
    # encoder(image with shape (BATCH,3,H,W)) -> z with shape (BATCH,latent_dim,h_small,w_small)
    # decoder(z)=reconstructed image with shape (BATCH,3,H,W)
    def __init__(self,encoder,decoder,embedding_dim,codebook_size=[256,128,128],embedding_scale=1):
        """
        Creates new vqvae2.
        encoder: 
            takes as input some tensor and returns three [(BATCH,embedding_dim,...)] tensors with encoded output for 3 quantization levels, high,middle and low
        decoder: 
            takes as input three [(BATCH,embedding_dim,...)] tensors and returns decoded tensor with original shape
        embedding_dim:
            channels count of encoder output
        codebook_size: list[int]
            What is codebook size for three layers, high,middle,low
        embedding_scale: desired norm of initialization vectors in codebooks
        """
        super().__init__()
        self.quantizer_high = VectorQuantizer(embedding_dim,codebook_size[0],0.99,1e-5,embedding_scale)
        self.quantizer_middle = VectorQuantizer(embedding_dim,codebook_size[1],0.99,1e-5,embedding_scale)
        self.quantizer_low = VectorQuantizer(embedding_dim,codebook_size[2],0.99,1e-5,embedding_scale)
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self,x):
        # something like
        # [64,32,32], [64,16,16]
        z_high,z_middle,z_low = self.encoder(x)
        # z = F.normalize(z, p=2.0, dim=1)

        # quantize low and high level features
        z_d_low,z_q_low,indices_low = self.quantizer_low(z_low)
        z_d_middle,z_q_middle,indices_middle = self.quantizer_middle(z_middle)
        z_d_high,z_q_high,indices_high = self.quantizer_high(z_high)

        x_rec = self.decoder([z_d_high,z_d_middle, z_d_low])
        # returns reconstruction, encoder outputs, quantized encoder outputs, indices of quantized vectors from codebooks
        return x_rec, [z_high,z_middle,z_low], [z_q_high,z_q_middle,z_q_low], [indices_high,indices_middle,indices_low]
    
def vqvae2_loss(x,x_rec,z,z_q,beta=0.25):
    """
    Computes loss for vqvae2 results.
    
    x: original x
    x_rec: reconstruction
    z: list of embeddings from encoder
    z_q: list of quantized embeddings
    beta: how fast to update encoder outputs z relative to reconstruction loss term
    """
    loss_ = lambda x,y : ((x-y)**2).mean()
    loss = 0
    # mse reconstruction loss
    loss += loss_(x,x_rec)
    
    # commitment loss
    for z_,z_q_ in zip(z,z_q):
        loss += beta*loss_(z_,z_q_.detach())/len(z)
    
    return loss