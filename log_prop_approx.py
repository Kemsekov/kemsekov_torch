import math
import torch
from torch.distributions import Normal

def generate_unit_simplex_vertices(d):
    # 1. Generate the initial regular simplex
    vertices = torch.eye(d)
    last_vertex = (1 - torch.sqrt(torch.tensor(d + 1.0))) / d * torch.ones(d)
    vertices = torch.cat([vertices, last_vertex.unsqueeze(0)], dim=0)
    
    # 2. Recenter at the origin (subtract the mean)
    vertices -= vertices.mean(dim=0)
    
    # 3. Normalize to unit length (radius = 1)
    # Each row is a vertex; normalize along the last dimension
    return torch.nn.functional.normalize(vertices, p=2, dim=-1)

def compute_subspace_log_volume(x: torch.Tensor, eps: float = 1e-8):
    """
    Computes the log-volume of the k-dimensional parallelepiped formed by 
    k vectors in N-dimensional space.
    """
    # 1. Transpose to [B, N, k] because QR decomposes columns
    # We want to find the volume spanned by the 'k' vectors.
    x_t = x.transpose(-1, -2)
    Q, R = torch.linalg.qr(x_t, mode='reduced')
    diag_r = torch.diagonal(R, dim1=-2, dim2=-1)
    log_vol = torch.sum(torch.log(torch.abs(diag_r) + eps), dim=-1)
    return log_vol

def log_prob(model, prior, eps=1e-3):
    """"
    Computes log_prob for density of random vector transformation y = model(prior) via jacobian approximation
    where y dimensions can differ from prior.
    
    model: model that transforms prior to target distribution
    prior: standard-normal distributed batched vector [...batch_dims...,dim]
    """
    data=prior
    device = 'cpu'
    Y = data.to(device)

    # generate N-dimensional simplex
    simplex_points = generate_unit_simplex_vertices(data.shape[-1]).to(device)*eps

    # simplex that have some point at origin 0
    shifted_simplex=simplex_points[:-1,:]-simplex_points[-1]

    # log area of original simplex
    original_simplex_area_log = shifted_simplex.slogdet()[1]
    # original_simplex_area_log = compute_subspace_log_volume(shifted_simplex)

    # make shapes match
    simplex_points = simplex_points.view(*([1]*(Y.ndim-1)),*simplex_points.shape)

    # shift Y to sphere points of simplex
    Y_neighbors = Y[...,None,:] + simplex_points  # (B, n_neighbors, ...dim)

    # Compute priors for all neighbors
    X_neighbors = model(Y_neighbors)

    # get area of transformed simplex
    transformed_simplex = X_neighbors[...,:-1,:]-X_neighbors[...,[-1],:]
    
    if transformed_simplex.shape[-1]==transformed_simplex.shape[-2]:
        transformed_simplex_area_log = transformed_simplex.slogdet()[1]
    else:
        transformed_simplex_area_log = compute_subspace_log_volume(transformed_simplex)

    in_dim = X_neighbors.shape[-1]
    
    # area ratio is our jacobian determinant approximation
    logdet_approx = transformed_simplex_area_log - original_simplex_area_log - in_dim*math.log(in_dim)

    prior_logp = Normal(0,1).log_prob(data).sum(-1)
    # this stuff perfectly match log prob structure
    return prior_logp-logdet_approx