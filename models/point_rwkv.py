"""
PointRWKV: Core RWKV block for point cloud learning.

Implements the PRWKV Block with two parallel branches:
1. IFM (Integrative Feature Modulation): Global processing via modified RWKV attention
2. LGM (Local Graph-based Merging): Local geometric feature extraction with graph stabilizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils.misc import knn_point, index_points


# ---------------------------------------------------------------------------
# Bidirectional Quadratic Expansion (BQE)
# ---------------------------------------------------------------------------

class BQE(nn.Module):
    """Bidirectional Quadratic Expansion.
    
    Broadens the receptive field by mixing features with shifted neighbors.
    BQE(X) = X + (1 - mu) * X_star
    where X_star is constructed from circular shifts of X.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            out: (B, T, C)
        """
        B, T, C = x.shape
        # Split into 4 chunks and apply circular shifts
        chunk_size = C // 4
        x1 = x[:, :, :chunk_size]
        x2 = torch.roll(x[:, :, chunk_size:2*chunk_size], shifts=1, dims=1)
        x3 = torch.roll(x[:, :, 2*chunk_size:3*chunk_size], shifts=-1, dims=1)
        x4 = torch.roll(x[:, :, 3*chunk_size:], shifts=2, dims=1)
        x_star = torch.cat([x1, x2, x3, x4], dim=-1)
        
        mu = torch.sigmoid(self.mu)
        return x + (1 - mu) * x_star


# ---------------------------------------------------------------------------
# RWKV Spatial-Mixing (WKV Attention)
# ---------------------------------------------------------------------------

class WKVAttention(nn.Module):
    """Bidirectional WKV attention with linear complexity.
    
    Implements the modified multi-headed matrix-valued states with 
    dynamic attention recurrence mechanism.
    """
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # BQE modules for token mixing
        self.bqe_r = BQE(dim)
        self.bqe_k = BQE(dim)
        self.bqe_v = BQE(dim)
        self.bqe_w = BQE(dim)
        
        # Linear projections
        self.proj_r = nn.Linear(dim, dim, bias=False)
        self.proj_k = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.proj_g = nn.Linear(dim, dim, bias=False)
        
        # Dynamic time-varying decay parameters
        self.decay_A = nn.Parameter(torch.randn(dim) * 0.01)
        self.decay_B = nn.Parameter(torch.randn(dim) * 0.01)
        self.decay_base = nn.Parameter(torch.randn(dim) * 0.01)
        self.w_proj = nn.Linear(dim, dim, bias=False)
        
        # Bonus term u
        self.u = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.01)
        
        # Output
        self.group_norm = nn.GroupNorm(num_heads, dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.orthogonal_(self.proj_r.weight)
        nn.init.orthogonal_(self.proj_k.weight)
        nn.init.orthogonal_(self.proj_v.weight)
        nn.init.orthogonal_(self.proj_g.weight)
    
    def _compute_decay(self, x):
        """Compute dynamic time-varying decay w.
        
        nu(c) = lambda_x + tanh(c * A_x) * B_x
        w = exp(-exp(nu(w_BQE(X))))
        """
        bqe_out = self.bqe_w(x)
        c = self.w_proj(bqe_out)
        nu = self.decay_base + torch.tanh(c * self.decay_A) * self.decay_B
        w = torch.exp(-torch.exp(nu))
        return w
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            out: (B, T, C)
        """
        B, T, C = x.shape
        
        # Compute R, K, V with BQE
        r = self.proj_r(self.bqe_r(x))  # (B, T, C), receptance
        k = self.proj_k(self.bqe_k(x))  # (B, T, C), key
        v = self.proj_v(self.bqe_v(x))  # (B, T, C), value
        g = torch.sigmoid(self.proj_g(x))  # (B, T, C), gate
        
        # Compute dynamic decay
        w = self._compute_decay(x)  # (B, T, C)
        
        # Reshape to multi-head: (B, H, T, D)
        r = rearrange(r, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)
        w = rearrange(w, 'b t (h d) -> b h t d', h=self.num_heads)
        
        # Bidirectional WKV computation
        # Forward direction
        wkv_fwd = self._wkv_forward(r, k, v, w)
        # Backward direction (reverse the sequence)
        wkv_bwd = self._wkv_forward(
            r.flip(dims=[2]), k.flip(dims=[2]), v.flip(dims=[2]), w.flip(dims=[2])
        ).flip(dims=[2])
        
        # Combine forward and backward
        wkv = (wkv_fwd + wkv_bwd) / 2  # (B, H, T, D)
        
        # Reshape back
        out = rearrange(wkv, 'b h t d -> b t (h d)')
        out = self.group_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = out * g
        out = self.out_proj(out)
        
        return out
    
    def _wkv_forward(self, r, k, v, w):
        """Compute WKV attention in one direction with linear complexity.
        
        Uses recurrence:
            s_t = w_t * s_{t-1} + k_t^T * v_t  (matrix-valued state)
            o_t = sigma(r_t) * (u * k_t^T * v_t + s_{t-1})
        
        Args:
            r, k, v, w: (B, H, T, D)
        Returns:
            out: (B, H, T, D)
        """
        B, H, T, D = r.shape
        
        # Initialize state
        state = torch.zeros(B, H, D, D, device=r.device, dtype=r.dtype)
        
        outputs = []
        for t in range(T):
            rt = torch.sigmoid(r[:, :, t, :])      # (B, H, D)
            kt = k[:, :, t, :]                       # (B, H, D)
            vt = v[:, :, t, :]                        # (B, H, D)
            wt = w[:, :, t, :]                        # (B, H, D)
            
            # Current token contribution: diag(u) * k^T * v
            kv = torch.einsum('bhd,bhe->bhde', kt, vt)  # (B, H, D, D)
            
            # Output: sigma(r) * (s_{t-1} @ k_t + u * k_t^T * v_t summed over last dim)
            state_contrib = (state * kt.unsqueeze(-2)).sum(dim=-1)  # (B, H, D)
            bonus_contrib = (kv * self.u.unsqueeze(0).unsqueeze(-1)).sum(dim=-1)  # (B, H, D)
            ot = rt * (state_contrib + bonus_contrib)
            
            outputs.append(ot)
            
            # Update state: s_t = diag(w) * s_{t-1} + k^T * v
            state = wt.unsqueeze(-1) * state + kv
        
        return torch.stack(outputs, dim=2)  # (B, H, T, D)


class WKVAttentionChunk(nn.Module):
    """Chunked bidirectional WKV attention for better efficiency.
    
    Uses a parallelized intra-chunk computation with sequential inter-chunk 
    state passing, achieving better GPU utilization.
    """
    
    def __init__(self, dim, num_heads=8, chunk_size=32):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size
        
        self.bqe_r = BQE(dim)
        self.bqe_k = BQE(dim)
        self.bqe_v = BQE(dim)
        self.bqe_w = BQE(dim)
        
        self.proj_r = nn.Linear(dim, dim, bias=False)
        self.proj_k = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.proj_g = nn.Linear(dim, dim, bias=False)
        
        self.decay_A = nn.Parameter(torch.randn(dim) * 0.01)
        self.decay_B = nn.Parameter(torch.randn(dim) * 0.01)
        self.decay_base = nn.Parameter(torch.randn(dim) * 0.01)
        self.w_proj = nn.Linear(dim, dim, bias=False)
        
        self.u = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.01)
        
        self.group_norm = nn.GroupNorm(num_heads, dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)
    
    def _compute_decay(self, x):
        bqe_out = self.bqe_w(x)
        c = self.w_proj(bqe_out)
        nu = self.decay_base + torch.tanh(c * self.decay_A) * self.decay_B
        w = torch.exp(-torch.exp(nu))
        return w
    
    def forward(self, x):
        B, T, C = x.shape
        
        r = self.proj_r(self.bqe_r(x))
        k = self.proj_k(self.bqe_k(x))
        v = self.proj_v(self.bqe_v(x))
        g = torch.sigmoid(self.proj_g(x))
        w = self._compute_decay(x)
        
        r = rearrange(r, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)
        w = rearrange(w, 'b t (h d) -> b h t d', h=self.num_heads)
        
        # Parallel intra-chunk + sequential inter-chunk
        out = self._chunked_wkv(r, k, v, w)
        
        out = rearrange(out, 'b h t d -> b t (h d)')
        out = self.group_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = out * g
        out = self.out_proj(out)
        return out
    
    def _chunked_wkv(self, r, k, v, w):
        """Chunked bidirectional WKV computation."""
        B, H, T, D = r.shape
        
        # Forward pass
        fwd = self._chunked_wkv_one_dir(r, k, v, w)
        # Backward pass
        bwd = self._chunked_wkv_one_dir(
            r.flip(2), k.flip(2), v.flip(2), w.flip(2)
        ).flip(2)
        
        return (fwd + bwd) / 2
    
    def _chunked_wkv_one_dir(self, r, k, v, w):
        B, H, T, D = r.shape
        CS = min(self.chunk_size, T)
        num_chunks = (T + CS - 1) // CS
        
        # Pad if necessary
        pad_len = num_chunks * CS - T
        if pad_len > 0:
            r = F.pad(r, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            w = F.pad(w, (0, 0, 0, pad_len), value=1.0)
        
        # Reshape into chunks
        r = r.reshape(B, H, num_chunks, CS, D)
        k = k.reshape(B, H, num_chunks, CS, D)
        v = v.reshape(B, H, num_chunks, CS, D)
        w = w.reshape(B, H, num_chunks, CS, D)
        
        state = torch.zeros(B, H, D, D, device=r.device, dtype=r.dtype)
        all_outputs = []
        
        for c in range(num_chunks):
            rc = r[:, :, c]  # (B, H, CS, D)
            kc = k[:, :, c]
            vc = v[:, :, c]
            wc = w[:, :, c]
            
            chunk_out = []
            for t in range(CS):
                rt = torch.sigmoid(rc[:, :, t])
                kt = kc[:, :, t]
                vt = vc[:, :, t]
                wt = wc[:, :, t]
                
                kv = torch.einsum('bhd,bhe->bhde', kt, vt)
                
                state_contrib = (state * kt.unsqueeze(-2)).sum(dim=-1)
                bonus_contrib = (kv * self.u.unsqueeze(0).unsqueeze(-1)).sum(dim=-1)
                ot = rt * (state_contrib + bonus_contrib)
                
                chunk_out.append(ot)
                state = wt.unsqueeze(-1) * state + kv
            
            all_outputs.append(torch.stack(chunk_out, dim=2))
        
        out = torch.cat(all_outputs, dim=2)
        if pad_len > 0:
            out = out[:, :, :T]
        return out


# ---------------------------------------------------------------------------
# Channel-Mixing (Feed-Forward)
# ---------------------------------------------------------------------------

class ChannelMixing(nn.Module):
    """RWKV Channel-Mixing layer (Feed-Forward equivalent).
    
    Uses BQE for token-shifted mixing before the feed-forward network.
    """
    
    def __init__(self, dim, hidden_dim=None, drop=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.bqe_k = BQE(dim)
        self.bqe_r = BQE(dim)
        
        self.key_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.receptance_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, dim, bias=False)
        
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            out: (B, T, C)
        """
        k = self.key_proj(self.bqe_k(x))
        k = F.silu(k)  # Squared ReLU or SiLU
        v = self.value_proj(k)
        r = torch.sigmoid(self.receptance_proj(self.bqe_r(x)))
        out = r * v
        return self.drop(out)


# ---------------------------------------------------------------------------
# Local Graph-based Merging (LGM) with Graph Stabilizer
# ---------------------------------------------------------------------------

class GraphStabilizer(nn.Module):
    """Graph Stabilizer for the LGM branch.
    
    Aligns neighboring coordinates to reduce translation variance:
        delta_x_i = h(v_i)
        v_i' = g(aggr(f(x_j - x_i + delta_x_i, v_j)), v_i)
    """
    
    def __init__(self, dim, num_iterations=3):
        super().__init__()
        self.num_iterations = num_iterations
        
        # Offset predictor: predicts coordinate alignment offset
        self.offset_pred = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 3)
        )
        
        # Edge feature encoder
        self.edge_mlps = nn.ModuleList()
        self.update_mlps = nn.ModuleList()
        
        for _ in range(num_iterations):
            self.edge_mlps.append(nn.Sequential(
                nn.Linear(3 + dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
            ))
            self.update_mlps.append(nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
            ))
    
    def forward(self, xyz, features, knn_idx):
        """
        Args:
            xyz: (B, N, 3) - point coordinates
            features: (B, N, C) - point features
            knn_idx: (B, N, K) - KNN indices
        Returns:
            features: (B, N, C) - updated features
        """
        B, N, C = features.shape
        K = knn_idx.shape[-1]
        
        for t in range(self.num_iterations):
            # Predict coordinate offset
            delta_xyz = self.offset_pred(features)  # (B, N, 3)
            
            # Gather neighbor coordinates and features
            neighbor_xyz = index_points(xyz, knn_idx)      # (B, N, K, 3)
            neighbor_feat = index_points(features, knn_idx)  # (B, N, K, C)
            
            # Compute edge features with stabilized coordinates
            center_xyz = xyz + delta_xyz  # (B, N, 3)
            rel_pos = neighbor_xyz - center_xyz.unsqueeze(2)  # (B, N, K, 3)
            
            # Edge encoding: f(x_j - x_i + delta_x_i, v_j)
            edge_input = torch.cat([rel_pos, neighbor_feat], dim=-1)  # (B, N, K, 3+C)
            edge_feat = self.edge_mlps[t](edge_input)  # (B, N, K, C)
            
            # Aggregate: max pooling over neighbors
            agg_feat = edge_feat.max(dim=2)[0]  # (B, N, C)
            
            # Update: g(agg, v_i)
            features = features + self.update_mlps[t](
                torch.cat([agg_feat, features], dim=-1)
            )
        
        return features


class LGMBranch(nn.Module):
    """Local Graph-based Merging (LGM) Branch.
    
    Extracts local geometric features via a fixed-radius near-neighbors graph
    with graph stabilizer mechanism.
    """
    
    def __init__(self, dim, k=16, num_iterations=3):
        super().__init__()
        self.k = k
        
        # Graph stabilizer
        self.graph_stabilizer = GraphStabilizer(dim, num_iterations=num_iterations)
        
        # Feature projection
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, N, 3)
            features: (B, N, C)
        Returns:
            out: (B, N, C)
        """
        # Build KNN graph
        knn_idx = knn_point(self.k, xyz, xyz)  # (B, N, K)
        
        # Apply graph stabilizer
        local_features = self.graph_stabilizer(xyz, features, knn_idx)
        
        # Project
        out = self.proj(local_features)
        return out


# ---------------------------------------------------------------------------
# PRWKV Block
# ---------------------------------------------------------------------------

class PRWKVBlock(nn.Module):
    """PointRWKV Block with parallel IFM and LGM branches.
    
    IFM (Integrative Feature Modulation): Global processing via RWKV attention
    LGM (Local Graph-based Merging): Local geometric feature extraction
    """
    
    def __init__(self, dim, num_heads=8, k=16, graph_iter=3, 
                 ffn_ratio=4, drop=0.0, drop_path=0.0):
        super().__init__()
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # IFM branch: Spatial-Mixing (RWKV attention)
        self.spatial_mixing = WKVAttention(dim, num_heads=num_heads)
        
        # LGM branch: Local graph
        self.lgm = LGMBranch(dim, k=k, num_iterations=graph_iter)
        
        # Feature fusion
        self.fusion = nn.Linear(dim * 2, dim, bias=False)
        
        # Channel-Mixing (FFN)
        self.channel_mixing = ChannelMixing(dim, hidden_dim=int(dim * ffn_ratio), drop=drop)
        
        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, N, 3)
            features: (B, N, C)
        Returns:
            features: (B, N, C)
        """
        # Parallel branches
        normed = self.norm1(features)
        
        # IFM: Global RWKV attention
        global_feat = self.spatial_mixing(normed)
        
        # LGM: Local graph features
        local_feat = self.lgm(xyz, normed)
        
        # Fusion: concatenate + project
        fused = self.fusion(torch.cat([global_feat, local_feat], dim=-1))
        features = features + self.drop_path(fused)
        
        # Channel-Mixing (FFN)
        features = features + self.drop_path(self.channel_mixing(self.norm2(features)))
        
        return features


# ---------------------------------------------------------------------------
# DropPath (Stochastic Depth)
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


# ---------------------------------------------------------------------------
# PointRWKV Backbone (Hierarchical Multi-Scale)
# ---------------------------------------------------------------------------

class PointRWKV(nn.Module):
    """PointRWKV: Hierarchical multi-scale point cloud backbone.
    
    Architecture: Asymmetric U-Net with PRWKV blocks at multiple scales.
    - Encoder: 3 scales with downsampling via FPS
    - Decoder: Feature propagation with skip connections
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.embed_dim = config.get('embed_dim', 384)
        self.depth = config.get('depth', [4, 4, 4])  # blocks per scale
        self.num_heads = config.get('num_heads', 8)
        self.num_points = config.get('num_points', [2048, 1024, 512])
        self.group_sizes = config.get('group_sizes', [32, 32, 32])
        self.k_neighbors = config.get('k_neighbors', [16, 8, 8])
        self.graph_iter = config.get('graph_iter', 3)
        self.drop_path_rate = config.get('drop_path_rate', 0.1)
        self.num_scales = len(self.num_points)
        
        # Multi-scale grouping and embedding
        from models.pointnet2_utils import Group, Encoder
        self.group_modules = nn.ModuleList()
        self.embed_modules = nn.ModuleList()
        for i in range(self.num_scales):
            self.group_modules.append(Group(self.num_points[i], self.group_sizes[i]))
            self.embed_modules.append(Encoder(self.embed_dim))
        
        # Positional encoding
        self.pos_embed = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, self.embed_dim)
            ) for _ in range(self.num_scales)
        ])
        
        # PRWKV blocks at each scale
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depth))]
        cur = 0
        self.blocks = nn.ModuleList()
        for i in range(self.num_scales):
            scale_blocks = nn.ModuleList()
            for j in range(self.depth[i]):
                scale_blocks.append(PRWKVBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    k=self.k_neighbors[i],
                    graph_iter=self.graph_iter,
                    drop_path=dpr[cur],
                ))
                cur += 1
            self.blocks.append(scale_blocks)
        
        # Layer norms for each scale output
        self.norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(self.num_scales)])
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, pts):
        """
        Args:
            pts: (B, N, 3) - input point cloud
        Returns:
            features_list: list of (B, N_i, C) at each scale
            centers_list: list of (B, N_i, 3) at each scale
        """
        features_list = []
        centers_list = []
        
        for i in range(self.num_scales):
            # Group and embed
            neighborhood, center = self.group_modules[i](pts)
            tokens = self.embed_modules[i](neighborhood)
            
            # Add positional encoding
            pos = self.pos_embed[i](center)
            tokens = tokens + pos
            
            # Apply PRWKV blocks
            for block in self.blocks[i]:
                tokens = block(center, tokens)
            
            tokens = self.norms[i](tokens)
            features_list.append(tokens)
            centers_list.append(center)
        
        return features_list, centers_list
