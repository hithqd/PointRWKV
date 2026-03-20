"""PointNet++ utility functions for point cloud processing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import fps, index_points, knn_point, square_distance


class PointNetFeaturePropagation(nn.Module):
    """Feature propagation (upsampling) module from PointNet++."""
    
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: (B, N, 3) - target positions (more points)
            xyz2: (B, S, 3) - source positions (fewer points)
            points1: (B, N, D1) - target features (skip connection)
            points2: (B, S, D2) - source features to propagate
        Returns:
            new_points: (B, N, D_out)
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)  # (B, N, S)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # 3 nearest neighbors

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # (B, N, 3)

            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.unsqueeze(-1), dim=2
            )  # (B, N, D2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)  # (B, D, N)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points.permute(0, 2, 1)  # (B, N, D_out)


class Group(nn.Module):
    """FPS + KNN grouping for creating point patches."""
    
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3)
        Returns:
            neighborhood: (B, G, K, 3), grouped and centered point patches
            center: (B, G, 3), center positions
        """
        B, N, C = xyz.shape
        # FPS to find center points
        center_idx = fps(xyz, self.num_group)  # (B, G)
        center = index_points(xyz, center_idx)  # (B, G, 3)
        
        # KNN to find neighbors for each center
        idx = knn_point(self.group_size, xyz, center)  # (B, G, K)
        neighborhood = index_points(xyz, idx)  # (B, G, K, 3)
        
        # Normalize to local coordinates
        neighborhood = neighborhood - center.unsqueeze(2)  # center the patches
        return neighborhood, center


class Encoder(nn.Module):
    """Mini-PointNet encoder for point patches."""
    
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1),
        )

    def forward(self, point_groups):
        """
        Args:
            point_groups: (B, G, K, 3)
        Returns:
            feature: (B, G, C)
        """
        B, G, K, _ = point_groups.shape
        # reshape to (B*G, 3, K)
        point_groups = point_groups.reshape(B * G, K, 3).permute(0, 2, 1)
        
        feature = self.first_conv(point_groups)  # (B*G, 256, K)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (B*G, 256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, K), feature], dim=1)  # (B*G, 512, K)
        feature = self.second_conv(feature)  # (B*G, C, K)
        feature_global = torch.max(feature, dim=2)[0]  # (B*G, C)
        return feature_global.reshape(B, G, self.encoder_channel)


class MultiScaleGrouping(nn.Module):
    """Multi-scale point cloud grouping for hierarchical feature learning."""
    
    def __init__(self, num_points_list, group_sizes, embed_dim):
        """
        Args:
            num_points_list: list of int, e.g. [2048, 1024, 512]
            group_sizes: list of int, e.g. [32, 32, 32]
            embed_dim: int, embedding dimension
        """
        super().__init__()
        self.num_scales = len(num_points_list)
        self.num_points_list = num_points_list
        self.group_sizes = group_sizes
        
        self.groupers = nn.ModuleList()
        self.encoders = nn.ModuleList()
        
        for i in range(self.num_scales):
            self.groupers.append(Group(num_points_list[i], group_sizes[i]))
            self.encoders.append(Encoder(embed_dim))

    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3)
        Returns:
            features_list: list of (B, G_i, C) for each scale
            centers_list: list of (B, G_i, 3) for each scale
        """
        features_list = []
        centers_list = []
        
        for i in range(self.num_scales):
            neighborhood, center = self.groupers[i](xyz)
            feature = self.encoders[i](neighborhood)
            features_list.append(feature)
            centers_list.append(center)
        
        return features_list, centers_list
