import torch
import torch.nn as nn

class LearnedPosEmb(nn.Module):
    """
    Learnable positional embedding module for sequences.
    
    Accepts tensors of shape [batch,seq_len,dim]

    This module adds a learnable positional embedding to the input tensor,
    allowing the model to encode the position of each token in a sequence.

    Parameters
    ----------
    dim : int
        The dimensionality of the input embeddings.
    max_length_size : int
        The maximum sequence length the module can support.

    Attributes
    ----------
    learned_position : torch.nn.Parameter
        A learnable tensor of shape (1, max_length_size, dim) representing
        the positional embeddings.

    Examples
    --------
    >>> emb = LearnedPosEmb(dim=512, max_length_size=1024)
    >>> x = torch.randn(8, 100, 512)  # (batch_size, seq_len, dim)
    >>> out = emb(x)
    >>> out.shape
    torch.Size([8, 100, 512])
    """

    def __init__(self, dim: int, max_length_size: int):
        """
        Learnable positional embedding module for sequences.
        
        Accepts tensors of shape [batch,seq_len,dim]

        This module adds a learnable positional embedding to the input tensor,
        allowing the model to encode the position of each token in a sequence.

        Parameters
        ----------
        dim : int
            The dimensionality of the input embeddings.
        max_length_size : int
            The maximum sequence length the module can support.

        Attributes
        ----------
        learned_position : torch.nn.Parameter
            A learnable tensor of shape (1, max_length_size, dim) representing
            the positional embeddings.

        Examples
        --------
        >>> emb = LearnedPosEmb(dim=512, max_length_size=1024)
        >>> x = torch.randn(8, 100, 512)  # (batch_size, seq_len, dim)
        >>> out = emb(x)
        >>> out.shape
        torch.Size([8, 100, 512])
        """
        super().__init__()
        self.learned_position = nn.Parameter(torch.zeros(1, max_length_size, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learnable positional embeddings to the input tensor.
    

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, dim).

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as input, with positional
            embeddings added.
        """
        seq_len = x.size(1)
        pos_emb = self.learned_position[:, :seq_len, :]
        return x + pos_emb
