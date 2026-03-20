"""PointRWKV Pre-training Model via Masked Autoencoding (MAE).

Pre-trains the PointRWKV backbone using a masked point cloud reconstruction
objective, following the Point-MAE / Point-M2AE paradigm.

Masking ratio: 80% (from paper ablation)
"""

import torch
import torch.nn as nn
import numpy as np
from models.point_rwkv import PRWKVBlock, DropPath
from models.pointnet2_utils import Group, Encoder


class MaskedAutoEncoder(nn.Module):
    """Masked Autoencoder for PointRWKV pre-training.
    
    Architecture:
    - Encoder: Processes only visible (unmasked) patches
    - Decoder: Reconstructs the masked patches from visible patch features + mask tokens
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        embed_dim = config.model.get('embed_dim', 384)
        decoder_dim = config.model.get('decoder_dim', 192)
        decoder_depth = config.model.get('decoder_depth', 4)
        num_heads = config.model.get('num_heads', 8)
        decoder_heads = config.model.get('decoder_heads', 4)
        group_size = config.model.get('group_sizes', [32, 32, 32])[0]
        num_group = config.model.get('num_points', [2048, 1024, 512])[0]
        mask_ratio = config.model.get('mask_ratio', 0.8)
        
        self.mask_ratio = mask_ratio
        self.num_group = num_group
        self.group_size = group_size
        self.embed_dim = embed_dim
        
        # Encoder components (operate on visible patches only)
        self.group = Group(num_group, group_size)
        self.encoder_embed = Encoder(embed_dim)
        
        # Positional encoding for encoder
        self.encoder_pos = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )
        
        # Encoder blocks (same as backbone's first scale)
        depth = config.model.get('depth', [4, 4, 4])
        encoder_depth = depth[0]
        k = config.model.get('k_neighbors', [16, 8, 8])[0]
        drop_path_rate = config.model.get('drop_path_rate', 0.1)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, encoder_depth)]
        
        self.encoder_blocks = nn.ModuleList([
            PRWKVBlock(
                dim=embed_dim,
                num_heads=num_heads,
                k=min(k, int(num_group * (1 - mask_ratio))),
                graph_iter=config.model.get('graph_iter', 3),
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        self.decoder_pos = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, decoder_dim)
        )
        
        # Simple transformer decoder (lightweight)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(decoder_dim, decoder_heads, drop_path=0.0)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        
        # Prediction head: reconstruct point coordinates
        self.pred_head = nn.Sequential(
            nn.Linear(decoder_dim, 256),
            nn.GELU(),
            nn.Linear(256, 3 * group_size)  # predict group_size * 3 coordinates
        )
        
        # Loss
        self.loss_type = config.model.get('loss_type', 'cdl2')
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _mask_center_block(self, center, noaug=False):
        """Generate mask for point patches.
        
        Args:
            center: (B, G, 3)
        Returns:
            bool_masked_pos: (B, G) - True for masked positions
        """
        B, G, _ = center.shape
        
        if noaug or self.mask_ratio == 0:
            return torch.zeros(B, G).bool().to(center.device)
        
        num_mask = int(self.mask_ratio * G)
        
        # Random masking
        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - num_mask),
                np.ones(num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        
        overall_mask = torch.from_numpy(overall_mask).bool().to(center.device)
        return overall_mask
    
    def forward(self, pts, noaug=False):
        """
        Args:
            pts: (B, N, 3)
        Returns:
            loss: reconstruction loss
        """
        # Group points into patches
        neighborhood, center = self.group(pts)  # (B, G, K, 3), (B, G, 3)
        
        # Embed patches
        tokens = self.encoder_embed(neighborhood)  # (B, G, C)
        
        # Generate mask
        bool_masked_pos = self._mask_center_block(center, noaug=noaug)  # (B, G)
        
        # Separate visible and masked
        B, G, C = tokens.shape
        
        vis_tokens = tokens[~bool_masked_pos].reshape(B, -1, C)
        vis_center = center[~bool_masked_pos].reshape(B, -1, 3)
        mask_center = center[bool_masked_pos].reshape(B, -1, 3)
        
        # Add position encoding to visible tokens
        vis_pos = self.encoder_pos(vis_center)
        vis_tokens = vis_tokens + vis_pos
        
        # Encode visible patches
        for block in self.encoder_blocks:
            vis_tokens = block(vis_center, vis_tokens)
        vis_tokens = self.encoder_norm(vis_tokens)
        
        # Decoder
        vis_tokens_dec = self.decoder_embed(vis_tokens)
        
        # Create mask tokens
        num_mask = mask_center.shape[1]
        mask_tokens = self.mask_token.expand(B, num_mask, -1)
        
        # Concatenate visible + mask tokens
        full_tokens = torch.cat([vis_tokens_dec, mask_tokens], dim=1)
        full_center = torch.cat([vis_center, mask_center], dim=1)
        full_pos = self.decoder_pos(full_center)
        full_tokens = full_tokens + full_pos
        
        # Decode
        for block in self.decoder_blocks:
            full_tokens = block(full_tokens)
        full_tokens = self.decoder_norm(full_tokens)
        
        # Predict only masked patches
        mask_tokens_out = full_tokens[:, -num_mask:]
        pred = self.pred_head(mask_tokens_out)  # (B, num_mask, group_size*3)
        pred = pred.reshape(B, num_mask, self.group_size, 3)
        
        # Get ground truth masked patches
        gt = neighborhood[bool_masked_pos].reshape(B, num_mask, self.group_size, 3)
        
        # Compute loss
        loss = self._compute_loss(pred, gt)
        
        return loss
    
    def _compute_loss(self, pred, gt):
        """Compute reconstruction loss (Chamfer Distance L2)."""
        # pred, gt: (B, M, K, 3)
        B, M, K, _ = pred.shape
        pred = pred.reshape(B * M, K, 3)
        gt = gt.reshape(B * M, K, 3)
        
        # Chamfer Distance L2
        dist1 = torch.cdist(pred, gt, p=2)  # (B*M, K, K)
        loss1 = dist1.min(dim=2)[0].mean(dim=1)  # pred -> gt
        loss2 = dist1.min(dim=1)[0].mean(dim=1)  # gt -> pred
        
        loss = (loss1 + loss2).mean()
        return loss


class DecoderBlock(nn.Module):
    """Simple Transformer decoder block for MAE decoder."""
    
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x):
        normed = self.norm1(x)
        x = x + self.drop_path(self.attn(normed, normed, normed)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# Alias
PointRWKVPretrain = MaskedAutoEncoder
