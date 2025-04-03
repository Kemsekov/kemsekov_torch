import math
from typing import List
import torch.nn as nn
from kemsekov_torch.vqvae.quantizer import *
from kemsekov_torch.residual import Residual, ResidualBlock
from kemsekov_torch.dpsa import DPSABlock
from kemsekov_torch.positional_emb import ConcatPositionalEmbeddingPermute
from kemsekov_torch.conv_modules import SCSEModule
from kemsekov_torch.common_modules import get_normalization_from_name

class VQVAE2Scale3(nn.Module):
    # encoder(image with shape (BATCH,3,H,W)) -> z with shape (BATCH,latent_dim,h_small,w_small)
    # decoder(z)=reconstructed image with shape (BATCH,3,H,W)
    def __init__(
        self,
        in_channels,
        embedding_dim,
        latent_dim,
        codebook_size=[256,256,256],
        compression_ratio=4,
        decay=0.99,
        epsilon=1e-5,
        dimensions=2,
        num_residual_layers=3,
        ):
        """
        Creates new vqvae2.
        
        in_channels: input tensor channels
        
        embedding_dim: dimensions for internal computation
        
        latent_dim: output latent space channels
        
        codebook_size: codebook size per scaling layer, `[bottom_codebook_size,mid_codebook_size,top_codebook_size]`. It is common to use larger codebooks for bottom and mid and use small codebook for top scales.
        
        compression_ratio: defines how much we want to compress input image, compression_ratio=4 with 256x256 image will be compressed to 64x64 latent tensor
        
        decay: decay for EMA update of codebook
        
        epsilon: used to update codebook
        
        dimensions: input dimension shape, so if we use dimensions=1 we can process sequence, if we use dimensions=2 we can process images, if we use dimensions=3 we can process video, etc
        
        num_residual_layers: how many residual layers to use
        """
        super().__init__()
        self.quantizer_bottom = VectorQuantizer(embedding_dim,codebook_size[0],decay,epsilon,1)
        self.quantizer_mid    = VectorQuantizer(embedding_dim,codebook_size[1],decay,epsilon,1)
        self.quantizer_top    = VectorQuantizer(embedding_dim,codebook_size[2],decay,epsilon,1)
        
        conv = [nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1]
        common = {
            "normalization":'batch',
            "dimensions":dimensions,
            'activation':torch.nn.SiLU
        }
        
        # make residual layers to be very cheap by not computing full emb_dim to emb_dim conv
        # but by using internal small dim like emb_dim -> res_dim -> emb_dim
        res_dim = embedding_dim//4
        
        # data compression over single stride 2 convolution
        compression_per_conv = 2**(dimensions)
        
        input_dim_expansion = [embedding_dim]*int(math.log2(compression_ratio))
        input_dim_expansion[0] = 2*in_channels*compression_per_conv
        for i in range(1,len(input_dim_expansion)):
            input_dim_expansion[i]=input_dim_expansion[i-1]*compression_per_conv
            input_dim_expansion[i]=min(embedding_dim,input_dim_expansion[i])
            
        input_dim_expansion[-1]=embedding_dim
        output_dim_expansion = list(reversed(input_dim_expansion))
        
        print("input_dim_expansion",input_dim_expansion)
        print("output_dim_expansion",output_dim_expansion)
        
        self.encoder_bottom = nn.Sequential(
            get_normalization_from_name(dimensions,common['normalization'])(in_channels),
            *[
                ResidualBlock(inp,outp,kernel_size=4,stride=2,normalization=common['normalization'],dimensions=dimensions, activation=common['activation'])
                for inp,outp in zip([in_channels]+input_dim_expansion,input_dim_expansion)
            ],
            
            *[ResidualBlock(embedding_dim,[res_dim,embedding_dim],**common) for i in range(num_residual_layers)],
        )
        
        # channels -> channels
        self.encoder_mid  = nn.Sequential(
            ResidualBlock(embedding_dim,embedding_dim,kernel_size=4,stride=2,**common),
            *[ResidualBlock(embedding_dim,[res_dim,embedding_dim],**common) for i in range(num_residual_layers)],
        )
        
        # channels -> channels
        self.encoder_top  = nn.Sequential(
            ResidualBlock(embedding_dim,embedding_dim,kernel_size=4,stride=2,**common),
            *[ResidualBlock(embedding_dim,[res_dim,embedding_dim],**common) for i in range(num_residual_layers)],
        )
        
        self.decoder_top = nn.Sequential(
            ResidualBlock(embedding_dim,[res_dim,embedding_dim],kernel_size=4,stride=2,**common).transpose(),
            *[ResidualBlock(embedding_dim,[res_dim,embedding_dim],**common) for i in range(num_residual_layers)],
            
        )
        
        self.decoder_mid = nn.Sequential(
            *[ResidualBlock(embedding_dim,[res_dim,embedding_dim],**common) for i in range(num_residual_layers)],
            ResidualBlock(embedding_dim,[res_dim,embedding_dim],kernel_size=4,stride=2,**common).transpose(),
        )
        
        self.combine_scales = nn.Sequential(
            ResidualBlock(3*embedding_dim,embedding_dim,**common),
            ResidualBlock(embedding_dim,latent_dim,**common)
        )
        
        self.decoder_bottom = nn.Sequential(
            ResidualBlock(latent_dim,embedding_dim,**common),
            *[ResidualBlock(embedding_dim,[res_dim,embedding_dim],**common) for i in range(num_residual_layers)],

            *[
                ResidualBlock(inp,outp,kernel_size=4,stride=2,**common).transpose()
                for inp,outp in zip([embedding_dim]+output_dim_expansion,output_dim_expansion)
            ],
            conv(output_dim_expansion[-1],in_channels,1)
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
        z = self.combine_scales(z)
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
        zd_bottom = F.normalize(zd_bottom, p=2.0, dim=1)
        
        zd_mid = self.quantizer_mid.decode_from_ind(mid)
        zd_mid = F.normalize(zd_mid, p=2.0, dim=1)        
        
        zd_top = self.quantizer_top.decode_from_ind(top)
        zd_top = F.normalize(zd_top, p=2.0, dim=1)        
        
        total_quant = [zd_bottom,self.upsample_mid(zd_mid),self.upsample_top(zd_top)]
        z = torch.concat(total_quant,1)
        z = self.combine_scales(z)
        
        return self.decoder_bottom(z)

class Discriminator(nn.Module):
    def __init__(self,in_channels,out_classes = 2,dimensions=2,normalization='spectral'):
        super().__init__()
        common = dict(
            kernel_size=4,
            stride=2,
            normalization=normalization,
            dimensions=dimensions
        )
        conv = [nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1]
        pool_to_1 = [nn.AdaptiveAvgPool1d,nn.AdaptiveAvgPool2d,nn.AdaptiveAvgPool3d][dimensions-1]([1]*dimensions)
        self.m = nn.Sequential(
            ResidualBlock(in_channels,[64,64],kernel_size=4,stride=4,normalization=normalization,dimensions=dimensions),
            ResidualBlock(64,128,**common),
            ResidualBlock(128,256,**common),
            ResidualBlock(256,512,dilation=[1]+[2]+[4],**common),
            pool_to_1,
            nn.Flatten(1),
            conv(512,out_classes,kernel_size=1)
            # nn.Linear(256,out_classes)
        )
    
    def forward(self,x):
        return self.m(x)


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
    
    # general mse reconstruction loss
    rec_loss = loss_(x,x_rec)
    
    commitment_loss = 0
    # commitment loss
    for z_,z_q_ in zip(z,z_q):
        commitment_loss += loss_(z_,z_q_.detach())/len(z)
    commitment_loss/=len(z)
    
    return rec_loss+beta*commitment_loss