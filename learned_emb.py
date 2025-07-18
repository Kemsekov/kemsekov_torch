import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
# i am sorry that this class is so ugly
# it is possible to implement it better, but it won't compile in that case

class LearnedPosEmb(nn.Module):
    """
    MLP-based positional embedding for arbitrary 1–4D spatial inputs.

    Adds a learned, continuous positional embedding to every spatial location.
    Works on tensors of shape (B, C, *spatial), where the number of spatial
    dimensions (len(spatial)) is given by `dimensions`.

    Parameters
    ----------
    dim : int
        Number of feature channels (C) in the input.
    dimensions : Literal[1,2,3,4]
        The number of spatial dimensions (1 to 4).
    hidden : int, optional (default = dim)
        Hidden size of the MLP that maps normalized coords → embeddings.
    """

    def __init__(
        self,
        dim: int,
        dimensions: Literal[1, 2, 3, 4],
        hidden: int = None,
    ):
        super().__init__()
        self.dimensions = dimensions
        self.dim = dim
        hidden = hidden or dim

        # an MLP that takes `dimensions` coords → `dim` output
        self.to_pos = nn.Sequential(
            nn.Linear(dimensions, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Tanh()
        )
        # add residual scaler
        # self.gamma = nn.Parameter(torch.tensor(0.1))
        
        # rescale inputs by this factor, so we are working not with 512x512 space, but wil 512/32 x 512/32 space
        self.scale_factor = 32
        self.gamma = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add continuous positional embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, D1, D2, ..., Dk) where k == self.dimensions.

        Returns
        -------
        torch.Tensor
            Same shape as `x`, with per-location embeddings added.
        """
        if x.dim() != 2 + self.dimensions:
            raise ValueError(f"Expected input of rank {2 + self.dimensions}, got {x.dim()}")

        coords = self._get_coords(x)
        pos = self.to_pos(coords)  # (N, C)

        # Reshape to spatial + channel layout
        spatial_shape = [x.size(i + 2) for i in range(self.dimensions)]
        C = x.size(1)
        pos = pos.view(spatial_shape + [C])

        # Permute to (C, D1, D2, ...) and broadcast
        perm = [self.dimensions] + list(range(self.dimensions))
        pos = pos.permute(perm).unsqueeze(0)

        return x + self.gamma*pos

    def _get_coords(self, x: torch.Tensor) -> torch.Tensor:
        if self.dimensions == 1:
            return self._get_coords_1d(x)
        elif self.dimensions == 2:
            return self._get_coords_2d(x)
        elif self.dimensions == 3:
            return self._get_coords_3d(x)
        elif self.dimensions == 4:
            return self._get_coords_4d(x)
        else:
            raise ValueError("dimensions must be between 1 and 4")

    def _get_coords_1d(self, x: torch.Tensor) -> torch.Tensor:
        D1 = x.size(2)
        g1 = torch.linspace(0, D1/self.scale_factor, D1, device=x.device, dtype=x.dtype)
        return g1.unsqueeze(1)  # (D1, 1)

    def _get_coords_2d(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.size(2), x.size(3)
        gh = torch.linspace(0, H/self.scale_factor, H, device=x.device, dtype=x.dtype).unsqueeze(1).expand(H, W)
        gw = torch.linspace(0, W/self.scale_factor, W, device=x.device, dtype=x.dtype).unsqueeze(0).expand(H, W)
        return torch.stack([gh, gw], dim=-1).view(-1, 2)  # (H*W, 2)

    def _get_coords_3d(self, x: torch.Tensor) -> torch.Tensor:
        D1, D2, D3 = x.size(2), x.size(3), x.size(4)
        g1 = torch.linspace(0, D1/self.scale_factor, D1, device=x.device, dtype=x.dtype).view(D1, 1, 1).expand(D1, D2, D3)
        g2 = torch.linspace(0, D2/self.scale_factor, D2, device=x.device, dtype=x.dtype).view(1, D2, 1).expand(D1, D2, D3)
        g3 = torch.linspace(0, D3/self.scale_factor, D3, device=x.device, dtype=x.dtype).view(1, 1, D3).expand(D1, D2, D3)
        return torch.stack([g1, g2, g3], dim=-1).view(-1, 3)

    def _get_coords_4d(self, x: torch.Tensor) -> torch.Tensor:
        D1, D2, D3, D4 = x.size(2), x.size(3), x.size(4), x.size(5)
        g1 = torch.linspace(0, D1/self.scale_factor, D1, device=x.device, dtype=x.dtype).view(D1, 1, 1, 1).expand(D1, D2, D3, D4)
        g2 = torch.linspace(0, D2/self.scale_factor, D2, device=x.device, dtype=x.dtype).view(1, D2, 1, 1).expand(D1, D2, D3, D4)
        g3 = torch.linspace(0, D3/self.scale_factor, D3, device=x.device, dtype=x.dtype).view(1, 1, D3, 1).expand(D1, D2, D3, D4)
        g4 = torch.linspace(0, D4/self.scale_factor, D4, device=x.device, dtype=x.dtype).view(1, 1, 1, D4).expand(D1, D2, D3, D4)
        return torch.stack([g1, g2, g3, g4], dim=-1).view(-1, 4)
