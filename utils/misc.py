import torch
import numpy as np
import random


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def square_distance(src, dst):
    """Calculate Euclidean distance between each two points.
    
    Args:
        src: (B, N, C)
        dst: (B, M, C)
    Returns:
        dist: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(k, xyz, new_xyz):
    """KNN search.
    
    Args:
        k: int, number of neighbors
        xyz: (B, N, 3), reference points
        new_xyz: (B, S, 3), query points
    Returns:
        idx: (B, S, k), indices of nearest neighbors
    """
    dist = square_distance(new_xyz, xyz)  # (B, S, N)
    _, idx = dist.topk(k, dim=-1, largest=False, sorted=False)
    return idx


def fps(xyz, npoint):
    """Farthest Point Sampling.
    
    Args:
        xyz: (B, N, 3), input point positions
        npoint: int, number of sampled points
    Returns:
        centroids: (B, npoint), indices of sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """Index points by given indices.
    
    Args:
        points: (B, N, C)
        idx: (B, S) or (B, S, K)
    Returns:
        new_points: (B, S, C) or (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points



class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
