"""PointRWKV Part Segmentation Model.

Supports ShapeNetPart segmentation with per-point part label prediction.
"""

import torch
import torch.nn as nn
from models.point_rwkv import PointRWKV
from models.pointnet2_utils import PointNetFeaturePropagation


class PointRWKVPartSeg(nn.Module):
    """PointRWKV for part segmentation (ShapeNetPart)."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Backbone
        self.backbone = PointRWKV(config.model)
        
        embed_dim = config.model.get('embed_dim', 384)
        num_part_classes = config.model.get('num_part_classes', 50)
        num_shape_classes = config.model.get('num_shape_classes', 16)
        
        # Shape category embedding (one-hot -> embedding)
        self.cls_embed = nn.Sequential(
            nn.Linear(num_shape_classes, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, embed_dim)
        )
        
        # Decoder: propagate features back to original points
        self.propagation_layers = nn.ModuleList()
        
        # From scale 2 (512) -> scale 1 (1024)
        self.propagation_layers.append(
            PointNetFeaturePropagation(embed_dim * 2, [embed_dim, embed_dim])
        )
        # From scale 1 (1024) -> scale 0 (2048)
        self.propagation_layers.append(
            PointNetFeaturePropagation(embed_dim * 2, [embed_dim, embed_dim])
        )
        # From scale 0 (2048) -> original points
        self.propagation_layers.append(
            PointNetFeaturePropagation(embed_dim + 3, [embed_dim, embed_dim])
        )
        
        # Segmentation head: per-point prediction
        self.seg_head = nn.Sequential(
            nn.Conv1d(embed_dim + embed_dim, 512, 1),  # features + cls_embed
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, num_part_classes, 1)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, pts, cls_label, seg_labels=None):
        """
        Args:
            pts: (B, N, 3) - input points
            cls_label: (B, num_shape_classes) - one-hot shape category
            seg_labels: (B, N) - per-point part labels (optional)
        Returns:
            dict with 'logits' (B, N, num_parts), 'loss'
        """
        B, N, _ = pts.shape
        
        # Get multi-scale features from backbone
        features_list, centers_list = self.backbone(pts)
        
        # Decode: propagate from coarse to fine
        # Scale 2 -> Scale 1
        feat = self.propagation_layers[0](
            centers_list[1], centers_list[2],
            features_list[1], features_list[2]
        )
        
        # Scale 1 -> Scale 0
        feat = self.propagation_layers[1](
            centers_list[0], centers_list[1],
            features_list[0], feat
        )
        
        # Scale 0 -> Original points
        feat = self.propagation_layers[2](
            pts, centers_list[0],
            pts, feat  # Use raw xyz as skip features for original points
        )
        
        # Add shape class embedding (broadcast to all points)
        cls_feat = self.cls_embed(cls_label.float())  # (B, C)
        cls_feat = cls_feat.unsqueeze(1).expand(-1, N, -1)  # (B, N, C)
        
        # Concatenate and predict
        combined = torch.cat([feat, cls_feat], dim=-1)  # (B, N, C+C)
        combined = combined.permute(0, 2, 1)  # (B, C+C, N)
        
        logits = self.seg_head(combined)  # (B, num_parts, N)
        logits = logits.permute(0, 2, 1)  # (B, N, num_parts)
        
        ret = {'logits': logits}
        
        if seg_labels is not None:
            logits_flat = logits.reshape(-1, logits.shape[-1])
            labels_flat = seg_labels.reshape(-1).long()
            ret['loss'] = self.loss_fn(logits_flat, labels_flat)
        
        return ret
    
    def get_loss(self, pts, cls_label, seg_labels):
        ret = self.forward(pts, cls_label, seg_labels)
        return ret['loss']
