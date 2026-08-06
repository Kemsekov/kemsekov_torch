"""
Approximate point matching via projection + rank alignment
(fast / random-projection / sliced variants).
"""
import torch


def match_approximate_fast(A, B):
    # A and B shape: [1024, 3]
    
    # 1. Project 3D points down to 1D scalar fields
    # (Summing channels works beautifully for structural alignment)
    proj_A = A.sum(dim=-1)
    proj_B = B.sum(dim=-1)
    
    # 2. Get sorting indices
    idx_A = torch.argsort(proj_A)
    idx_B = torch.argsort(proj_B)
    
    # 3. Map A's sorted ranking into B's spatial ranking
    # Construct an inverse mapping for B
    inv_idx_B = torch.argsort(idx_B)
    A_matched = A[idx_A][inv_idx_B]
    
    return A_matched
def match_approximate_random_proj(A, B):
    # A, B shape: [1024, 3]
    device = A.device
    
    # 1. Generate a random unit vector in 3D space
    # This prevents the skewing/bias of a fixed axis
    rand_proj = torch.randn(A.shape[-1], 1, device=device)
    rand_proj = rand_proj / torch.norm(rand_proj)
    
    # 2. Project points down to 1D via matrix multiplication
    proj_A = (A @ rand_proj).squeeze(-1)  # Shape: [1024]
    proj_B = (B @ rand_proj).squeeze(-1)  # Shape: [1024]
    
    # 3. Fast sort alignment
    idx_A = torch.argsort(proj_A)
    idx_B = torch.argsort(proj_B)
    
    inv_idx_B = torch.argsort(idx_B)
    A_matched = A[idx_A][inv_idx_B]
    
    return A_matched
def match_approximate_sliced(A, B, num_projections=4):
    # A, B shape: [1024, 3]
    device = A.device
    N = A.shape[0]
    
    # 1. Sample multiple random projection vectors
    # Shape: [3, num_projections]
    projections = torch.randn(A.shape[-1], num_projections, device=device)
    projections = projections / torch.norm(projections, dim=0, keepdim=True)
    
    # 2. Project data: Shape [1024, num_projections]
    proj_A = A @ projections
    proj_B = B @ projections
    
    # 3. Sort along each projection axis independently
    idx_A = torch.argsort(proj_A, dim=0)
    idx_B = torch.argsort(proj_B, dim=0)
    inv_idx_B = torch.argsort(idx_B, dim=0)
    
    # 4. Use the first random projection axis' mapping as a base,
    # or randomly pick one projection per batch to keep code clean.
    # For extreme speed, just pick the first slice of the random stack:
    chosen_slice = torch.randint(0, num_projections, (1,)).item()
    
    A_matched = A[idx_A[:, chosen_slice]][inv_idx_B[:, chosen_slice]]
    return A_matched