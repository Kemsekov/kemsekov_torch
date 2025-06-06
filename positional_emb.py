from typing import Literal
import numpy as np
import torch
import torch.nn as nn
from kemsekov_torch.common_modules import get_normalization_from_name
class ConcatPositionalEmbeddingPermute(torch.nn.Module):
    """
    Concat input with shape (batch_size, ch, ...N dimensions...) to positional embedding
    """
    def __init__(self,channels,freq=1000,dimensions=2):
        """
        channels: input channels
        freq: embedding frequency, must equal to around input size
        """
        super().__init__()
        conv = [nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1]
        self.m = conv(2*channels,channels,kernel_size=1)
        self.emb = PositionalEncodingPermute(channels,freq=freq)
        # self.gamma = torch.nn.Parameter(torch.tensor(0.0))
        
    def forward(self,x):
        return self.m(torch.concat([x,self.emb(x)],1))

class AddPositionalEmbeddingPermute(torch.nn.Module):
    """
    Adds input with shape (batch_size, ch, ...N dimensions...) to positional embedding
    """
    def __init__(self,channels,freq=1000,dimensions=2):
        """
        channels: input channels
        freq: embedding frequency, must equal to around input size
        """
        super().__init__()
        self.emb = PositionalEncodingPermute(channels,freq=freq)
        self.gamma = torch.nn.Parameter(torch.tensor(0.0))
        
    def forward(self,x):
        return self.emb(x)+x

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding(nn.Module):
    """
    (batch_size, ...N dimensions..., ch)
    """
    def __init__(self, channels, dtype_override = None,freq=10000):
        """
        (batch_size, ...N dimensions..., ch)
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        :param freq: Embedding frequency. Must be around the same as average input length
        """
        super().__init__()
        self.positional_encodings = nn.ModuleList([
            PositionalEncoding1D(channels,dtype_override,freq),
            PositionalEncoding2D(channels,dtype_override,freq),
            PositionalEncoding3D(channels,dtype_override,freq)
        ])
    def forward(self,tensor : torch.Tensor):
        """
        :param tensor: A Nd tensor of size (batch_size, ...N dimensions..., ch)
        :return: Positional Encoding Matrix of size (batch_size, ...N dimensions..., ch)
        """
        ind = len(tensor.shape)-3
        assert ind in [0,1,2], "tensor.shape must have from 1 to 3 spacial dimensions with shape (batch_size, ...N dimensions..., ch)"
        
        if ind == 0:return self.positional_encodings[0](tensor)
        if ind == 1:return self.positional_encodings[1](tensor)
        if ind == 2:return self.positional_encodings[2](tensor)
        raise RuntimeError("error")

class PositionalEncodingPermute(nn.Module):
    """
    (batch_size, ch, ...N dimensions...)
    """
    def __init__(self, channels, dtype_override = None,freq=10000):
        """
        Accepts (batchsize, ch, ...dimensions...)
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        :param freq: Embedding frequency. Must be around the same as average input length
        """
        super().__init__()
        self.positional_encodings = nn.ModuleList([
            PositionalEncodingPermute1D(channels,dtype_override,freq),
            PositionalEncodingPermute2D(channels,dtype_override,freq),
            PositionalEncodingPermute3D(channels,dtype_override,freq)
        ])
    def forward(self,tensor : torch.Tensor) -> torch.Tensor:
        """
        :param tensor: A Nd tensor of size (batch_size, ch, ...N dimensions...)
        :return: Positional Encoding Matrix of size (batch_size, ch, ...N dimensions...)
        """
        ind = len(tensor.shape)-3
        assert ind in [0,1,2], "tensor.shape must have from 1 to 3 spacial dimensions with shape (batchsize, ch, ...N dimensions...)"
        
        if ind == 0:return self.positional_encodings[0](tensor)
        if ind == 1:return self.positional_encodings[1](tensor)
        if ind == 2:return self.positional_encodings[2](tensor)
        raise RuntimeError("error")

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels, dtype_override=None,freq=10000):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        inv_freq = 1.0 / (freq ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.channels = channels
        self.dtype_override = dtype_override

    def forward(self, tensor : torch.Tensor) -> torch.Tensor:
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros(
            (x, self.channels),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )
        emb[:, : self.channels] = emb_x
        return emb[None, :, :orig_ch].repeat(batch_size, 1, 1)


class PositionalEncodingPermute1D(nn.Module):
    def __init__(self, channels, dtype_override=None,freq=10000):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)
        """
        super(PositionalEncodingPermute1D, self).__init__()
        self.penc = PositionalEncoding1D(channels, dtype_override,freq)

    def forward(self, tensor : torch.Tensor) -> torch.Tensor:
        tensor = tensor.permute(0, 2, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 2, 1)

    @property
    def org_channels(self):
        return self.penc.org_channels



class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, dtype_override=None,freq=10000):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        inv_freq = 1.0 / (freq ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.dtype_override = dtype_override
        self.channels = channels

    def forward(self, tensor : torch.Tensor) -> torch.Tensor:
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros(
            (x, y, self.channels * 2),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels, dtype_override=None,freq=10000):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels, dtype_override,freq)

    def forward(self, tensor : torch.Tensor) -> torch.Tensor:
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels, dtype_override=None,freq=10000):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        inv_freq = 1.0 / (freq ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.dtype_override = dtype_override
        self.channels = channels

    def forward(self, tensor : torch.Tensor) -> torch.Tensor:
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_z = torch.arange(z, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros(
            (x, y, z, self.channels * 3),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        return emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)


class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels, dtype_override=None,freq=10000):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels, dtype_override,freq)

    def forward(self, tensor : torch.Tensor) -> torch.Tensor:
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels
