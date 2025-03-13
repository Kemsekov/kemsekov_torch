from typing import List
import torch.nn as nn
from kemsekov_torch.vqvae.quantizer import *
from kemsekov_torch.residual import ResidualBlock
from kemsekov_torch.conv_modules import SCSEModule

class VQVAE2Scale3(nn.Module):
    # encoder(image with shape (BATCH,3,H,W)) -> z with shape (BATCH,latent_dim,h_small,w_small)
    # decoder(z)=reconstructed image with shape (BATCH,3,H,W)
    def __init__(self,embedding_dim,codebook_size=[256,256,256],embedding_scale=1,decay=0.99,epsilon=1e-5):
        """
        Creates new vqvae2.
        embedding_dim:
            channels count of encoder output
        codebook_size: list[int]
            What is codebook size for three layers, high,middle,low
        embedding_scale: desired norm of initialization vectors in codebooks
        """
        super().__init__()
        self.quantizer_bottom = VectorQuantizer(embedding_dim,codebook_size[0],decay,epsilon,embedding_scale)
        self.quantizer_mid    = VectorQuantizer(embedding_dim,codebook_size[1],decay,epsilon,embedding_scale)
        self.quantizer_top    = VectorQuantizer(embedding_dim,codebook_size[2],decay,epsilon,embedding_scale)
        
        dimensions=2
        conv = [nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1]
        common = {
            "normalization":'batch',
            'x_residual_type':'conv'
        }
        res_dim = embedding_dim//2
        
        # input_ch -> channels
        self.encoder_bottom = nn.Sequential(
            ResidualBlock(3,[embedding_dim,embedding_dim],kernel_size=4,stride=4,**common),
            SCSEModule(embedding_dim),
            ResidualBlock(embedding_dim,[res_dim,embedding_dim],**common),
            ResidualBlock(embedding_dim,[res_dim,embedding_dim],**common),
        )
        # channels -> channels
        self.encoder_mid  = nn.Sequential(
            ResidualBlock(embedding_dim,embedding_dim,kernel_size=4,stride=2,**common),
            SCSEModule(embedding_dim),
            ResidualBlock(embedding_dim,[res_dim,embedding_dim],**common),
        )
        
        # channels -> channels
        self.encoder_top  = nn.Sequential(
            ResidualBlock(embedding_dim,embedding_dim,kernel_size=4,stride=2,**common),
            SCSEModule(embedding_dim),
            ResidualBlock(embedding_dim,[res_dim,embedding_dim],**common),
        )
        
        self.decoder_top = nn.Sequential(
            ResidualBlock(embedding_dim,[res_dim,embedding_dim],kernel_size=4,stride=2,**common).transpose(),
            SCSEModule(embedding_dim),
            ResidualBlock(embedding_dim,[res_dim,embedding_dim],**common),
        )
        
        self.decoder_mid = nn.Sequential(
            ResidualBlock(embedding_dim,[res_dim,embedding_dim],kernel_size=4,stride=2,**common).transpose(),
            SCSEModule(embedding_dim),
            ResidualBlock(embedding_dim,[res_dim,embedding_dim],**common),
        )
        
        self.decoder_bottom = nn.Sequential(
            ResidualBlock(3*embedding_dim,[res_dim,embedding_dim],**common),
            SCSEModule(embedding_dim),
            ResidualBlock(embedding_dim,[res_dim,embedding_dim],**common),
            ResidualBlock(embedding_dim,[embedding_dim,embedding_dim],kernel_size=4,stride=4,**common).transpose(),
            SCSEModule(embedding_dim),
            conv(embedding_dim,3,1)
        )
        self.combine_bottom_and_decode_mid = ResidualBlock(2*embedding_dim,embedding_dim,**common)
        self.combine_mid_and_decode_top = ResidualBlock(2*embedding_dim,embedding_dim,**common)
        self.upsample_mid = ResidualBlock(embedding_dim,embedding_dim,kernel_size=4,stride=2,**common).transpose()
        self.upsample_top = ResidualBlock(embedding_dim,[embedding_dim,embedding_dim],kernel_size=4,stride=4,**common).transpose()
        
    def forward(self,x):
        z, z_layers, zq_layers, indices_layers = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z_layers, zq_layers, indices_layers

    @torch.jit.export
    def decode(self,z):
        return self.decoder_bottom(z)
    
    @torch.jit.export
    def encode(self,x):
        all_z_emb = []
        all_zd_emb = []
        all_ind = []
        
        # something like
        z_bottom = self.encoder_bottom(x) #emb
        z_mid = self.encoder_mid(z_bottom) #emb
        z_top = self.encoder_top(z_mid) #emb
        z_top = F.normalize(z_top, p=2.0, dim=1)
        
        all_z_emb.append(z_top)
        zd_top,indices_top = self.quantizer_top(z_top) #emb
        zd_top = F.normalize(zd_top, p=2.0, dim=1)        
        dec_top = self.decoder_top(zd_top) # emb
        z_mid = torch.concat([z_mid,dec_top],1) # 2 emb
        z_mid = self.combine_mid_and_decode_top(z_mid) # emb
        z_mid = F.normalize(z_mid, p=2.0, dim=1)
        
        all_z_emb.append(z_mid)
        all_zd_emb.append(zd_top)
        all_ind.append(indices_top)
        
        zd_mid,indices_mid = self.quantizer_mid(z_mid)
        zd_mid = F.normalize(zd_mid, p=2.0, dim=1)        
        dec_mid = self.decoder_mid(zd_mid)
        z_bottom = torch.concat([z_bottom,dec_mid],1) # 2 emb
        z_bottom = self.combine_bottom_and_decode_mid(z_bottom)
        z_bottom=F.normalize(z_bottom, p=2.0, dim=1)
        
        all_z_emb.append(z_bottom)
        all_zd_emb.append(zd_mid)
        all_ind.append(indices_mid)
        
        zd_bottom,indices_bottom = self.quantizer_bottom(z_bottom)
        zd_bottom = F.normalize(zd_bottom, p=2.0, dim=1)        
        total_quant = [zd_bottom,self.upsample_mid(zd_mid),self.upsample_top(zd_top)]
        z = torch.concat(total_quant,1)
        # z=F.normalize(z, p=2.0, dim=1)

        all_zd_emb.append(zd_bottom)
        all_ind.append(indices_bottom)
        
        all_ind.reverse()
        all_zd_emb.reverse()
        all_z_emb.reverse()
        
        return z, all_z_emb, all_zd_emb, all_ind
    
    @torch.jit.export
    def decode_from_ind(self,all_ind : List[torch.Tensor]):
        bottom,mid,top = all_ind[0],all_ind[1],all_ind[2]
        zd_bottom = self.quantizer_bottom.decode_from_ind(bottom)
        zd_mid = self.quantizer_mid.decode_from_ind(mid)
        zd_top = self.quantizer_top.decode_from_ind(top)
        total_quant = [zd_bottom,self.upsample_mid(zd_mid),self.upsample_top(zd_top)]
        z = torch.concat(total_quant,1)
        return self.decoder_bottom(z)

import torchvision.transforms as T
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
    
    rec_loss = 0
    # general mse reconstruction loss
    rec_loss = 3*loss_(x,x_rec)
    for sigma in [0.1,0.5,1]:
        xgb = T.GaussianBlur(7,sigma)(x)
        rec_loss+=loss_(x-xgb,x_rec-xgb)
    rec_loss/=6
    
    commitment_loss = 0
    # commitment loss
    for z_,z_q_ in zip(z,z_q):
        commitment_loss += loss_(z_,z_q_.detach())/len(z)
    commitment_loss/=len(z)
    
    return rec_loss+beta*commitment_loss