"""PointRWKV Classification Model.

Supports:
- ModelNet40 classification
- ScanObjectNN classification
- Few-shot classification on ModelNet40
"""

import torch
import torch.nn as nn
from models.point_rwkv import PointRWKV


class PointRWKVCls(nn.Module):
    """PointRWKV for 3D shape classification."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Backbone
        self.backbone = PointRWKV(config.model)
        
        embed_dim = config.model.get('embed_dim', 384)
        num_classes = config.model.get('num_classes', 40)
        
        # Classification head
        # Aggregate multi-scale features
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.2))
    
    def forward(self, pts, labels=None):
        """
        Args:
            pts: (B, N, 3)
            labels: (B,) optional, for computing loss
        Returns:
            dict with 'logits', 'loss' (if labels provided)
        """
        features_list, centers_list = self.backbone(pts)
        
        # Global feature aggregation from multi-scale
        # Use max pooling and average pooling from the finest and coarsest scales
        feat_fine = features_list[0]   # (B, N1, C)
        feat_coarse = features_list[-1]  # (B, N3, C)
        
        # Global pooling
        feat_fine_max = feat_fine.max(dim=1)[0]      # (B, C)
        feat_coarse_max = feat_coarse.max(dim=1)[0]  # (B, C)
        
        # Concatenate multi-scale features
        global_feat = torch.cat([feat_fine_max, feat_coarse_max], dim=-1)  # (B, 2C)
        
        logits = self.head(global_feat)  # (B, num_classes)
        
        ret = {'logits': logits}
        if labels is not None:
            ret['loss'] = self.loss_fn(logits, labels)
        
        return ret
    
    def get_loss(self, pts, labels):
        ret = self.forward(pts, labels)
        return ret['loss']
    
    def get_acc(self, pts, labels):
        ret = self.forward(pts, labels)
        preds = ret['logits'].argmax(dim=-1)
        acc = (preds == labels).float().mean()
        return acc, ret['loss']


class PointRWKVClsVoting(nn.Module):
    """PointRWKV classification with voting strategy for better accuracy."""
    
    def __init__(self, config):
        super().__init__()
        self.model = PointRWKVCls(config)
        self.num_votes = config.get('num_votes', 10)
    
    @torch.no_grad()
    def forward(self, pts):
        """Voting inference: run multiple augmented versions."""
        all_logits = []
        for _ in range(self.num_votes):
            # Random scaling
            scale = torch.FloatTensor(1).uniform_(0.9, 1.1).to(pts.device)
            pts_aug = pts * scale
            
            ret = self.model(pts_aug)
            all_logits.append(ret['logits'])
        
        # Average logits
        logits = torch.stack(all_logits, dim=0).mean(dim=0)
        return {'logits': logits}
