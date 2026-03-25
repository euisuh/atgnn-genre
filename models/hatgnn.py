"""
H-ATGNN: Hierarchical Audio Tagging Graph Neural Network
Extends ATGNN (Singh et al., 2024) with:
  - Text LM label embedding initialisation
  - Hierarchical label DAG (mood -> genre -> sub-genre)
  - Cross-modal CLAP fusion
  - Hierarchical PLG/LLG with level-wise message passing
  - Consistency-penalised hierarchical loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_dense_adj
import numpy as np


# ---------------------------------------------------------------------------
# Graph convolution (max-relative, from ATGNN / DeepGCN)
# ---------------------------------------------------------------------------

class MaxRelativeGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_update = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, edge_index):
        """
        x          : (N, C)
        edge_index : (2, E)  [src, dst] pairs
        """
        src, dst = edge_index
        # For each dst node, gather all src neighbours and take max of (src - dst)
        diff = x[src] - x[dst]                          # (E, C)
        # Scatter max over each destination node
        max_diff = torch.zeros_like(x)
        # manual scatter_max (avoids torch_scatter dependency)
        for i in range(x.size(0)):
            mask = dst == i
            if mask.any():
                max_diff[i] = diff[mask].max(dim=0).values

        x_cat = torch.cat([x, max_diff], dim=-1)        # (N, 2C)
        return self.W_update(x_cat)                      # (N, out_dim)


class PGNBlock(nn.Module):
    """Single Patch GNN block: GraphConv + FFN + residual."""

    def __init__(self, dim, k=9, dilation=1, ffn_ratio=4):
        super().__init__()
        self.k = k
        self.dilation = dilation
        self.W_in  = nn.Linear(dim, dim)
        self.W_out = nn.Linear(dim, dim)
        self.graph_conv = MaxRelativeGraphConv(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio),
            nn.GELU(),
            nn.Linear(dim * ffn_ratio, dim),
        )

    def build_knn_graph(self, x):
        # x: (N, C) — build k-NN graph in feature space
        # torch_geometric knn_graph expects (N, C) and returns edge_index (2, E)
        edge_index = knn_graph(x.detach(), k=self.k * self.dilation,
                               loop=False, flow='target_to_source')
        if self.dilation > 1:
            # keep only every d-th neighbour (dilated aggregation)
            src, dst = edge_index
            keep = torch.arange(0, edge_index.size(1), self.dilation,
                                device=x.device)
            edge_index = edge_index[:, keep]
        return edge_index

    def forward(self, x):
        # x: (N, C)
        edge_index = self.build_knn_graph(x)
        h = self.W_in(x)
        h = self.graph_conv(h, edge_index)
        h = self.W_out(h)
        x = self.norm1(x + h)
        x = self.norm2(x + self.ffn(x))
        return x


# ---------------------------------------------------------------------------
# Hierarchical PLG: patch -> mood -> genre -> sub-genre
# ---------------------------------------------------------------------------

class HierarchicalPLGBlock(nn.Module):
    """
    Level-wise patch-label graph convolution.
    Mood nodes aggregate from patches.
    Genre nodes aggregate from patches + mood nodes.
    Sub-genre nodes aggregate from patches + mood + genre nodes.
    """

    def __init__(self, patch_dim, label_dim, n_mood, n_genre, n_subgenre,
                 k_plg=9):
        super().__init__()
        self.k = k_plg
        self.n_mood = n_mood
        self.n_genre = n_genre
        self.n_sub = n_subgenre

        # Per-level update projections
        self.W_mood   = nn.Linear(label_dim + patch_dim,  label_dim)
        self.W_genre  = nn.Linear(label_dim + patch_dim + label_dim, label_dim)
        self.W_sub    = nn.Linear(label_dim + patch_dim + label_dim + label_dim, label_dim)

        self.norm_mood  = nn.LayerNorm(label_dim)
        self.norm_genre = nn.LayerNorm(label_dim)
        self.norm_sub   = nn.LayerNorm(label_dim)

    def _label_to_patch_agg(self, label_emb, patch_emb, k):
        """
        For each label node, find k nearest patch nodes and aggregate via max-relative.
        label_emb : (S, C)
        patch_emb : (N, C)
        returns   : (S, C) aggregated context from patches
        """
        # Euclidean distances: (S, N)
        dists = torch.cdist(label_emb, patch_emb)      # (S, N)
        topk  = dists.topk(k, dim=-1, largest=False).indices  # (S, k)

        # Gather patch neighbours
        neighbours = patch_emb[topk]                   # (S, k, C)
        diff = neighbours - label_emb.unsqueeze(1)     # (S, k, C)
        max_diff = diff.max(dim=1).values              # (S, C)
        return max_diff

    def forward(self, patch_emb, mood_emb, genre_emb, sub_emb):
        # --- Level 1: mood from patches ---
        ctx_m = self._label_to_patch_agg(mood_emb, patch_emb, self.k)
        mood_emb = self.norm_mood(mood_emb + self.W_mood(
            torch.cat([mood_emb, ctx_m], dim=-1)))

        # --- Level 2: genre from patches + mood ---
        ctx_g_patch = self._label_to_patch_agg(genre_emb, patch_emb, self.k)
        ctx_g_mood  = self._label_to_patch_agg(genre_emb, mood_emb, max(1, self.k // 2))
        genre_emb = self.norm_genre(genre_emb + self.W_genre(
            torch.cat([genre_emb, ctx_g_patch, ctx_g_mood], dim=-1)))

        # --- Level 3: sub-genre from patches + mood + genre ---
        ctx_s_patch = self._label_to_patch_agg(sub_emb, patch_emb, self.k)
        ctx_s_mood  = self._label_to_patch_agg(sub_emb, mood_emb,  max(1, self.k // 3))
        ctx_s_genre = self._label_to_patch_agg(sub_emb, genre_emb, max(1, self.k // 2))
        sub_emb = self.norm_sub(sub_emb + self.W_sub(
            torch.cat([sub_emb, ctx_s_patch, ctx_s_mood, ctx_s_genre], dim=-1)))

        return mood_emb, genre_emb, sub_emb


# ---------------------------------------------------------------------------
# Hierarchical LLG: learnable adjacency, masked by DAG edges
# ---------------------------------------------------------------------------

class HierarchicalLLGBlock(nn.Module):
    """
    Label-Label GNN with adjacency matrix constrained by hierarchy topology.
    Within-level edges are fully learnable.
    Cross-level edges follow the DAG direction only.
    """

    def __init__(self, n_labels, label_dim, hierarchy_mask):
        """
        hierarchy_mask : (n_labels, n_labels) binary tensor
                         1 = edge is allowed (within-level or parent->child)
                         0 = edge is forbidden
        """
        super().__init__()
        self.register_buffer('mask', hierarchy_mask.float())
        # Raw adjacency (will be masked before use)
        self.A_raw = nn.Parameter(torch.randn(n_labels, n_labels) * 0.01)
        self.norm  = nn.LayerNorm(label_dim)

    def forward(self, L):
        """L: (S, C) concatenated mood+genre+sub label embeddings"""
        A = torch.sigmoid(self.A_raw) * self.mask   # (S, S) masked soft adjacency
        L_out = A @ L + L                            # residual aggregation
        return self.norm(L_out)


# ---------------------------------------------------------------------------
# Cross-modal gated fusion
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """
    Fuses patch graph features with CLAP audio-text embedding.
    Gate is learned per-dimension.
    """

    def __init__(self, patch_dim, clap_dim, out_dim):
        super().__init__()
        self.proj_patch = nn.Linear(patch_dim, out_dim)
        self.proj_clap  = nn.Linear(clap_dim,  out_dim)
        self.gate       = nn.Linear(out_dim * 2, out_dim)

    def forward(self, patch_feat, clap_feat):
        """
        patch_feat : (N, patch_dim) — flattened global avg of patch nodes
        clap_feat  : (B, clap_dim) — CLAP audio embedding
        """
        p = self.proj_patch(patch_feat)               # (B, D)
        c = self.proj_clap(clap_feat)                 # (B, D)
        g = torch.sigmoid(self.gate(torch.cat([p, c], dim=-1)))
        return g * p + (1 - g) * c                    # (B, D)


# ---------------------------------------------------------------------------
# CNN Backbone (lightweight, ImageNet-pretrained weights loadable)
# ---------------------------------------------------------------------------

class CNNBackbone(nn.Module):
    """
    Lightweight CNN backbone.
    In practice replace with EfficientNet-B2 or ResNet-50 for best results.
    This is a minimal version for quick experiments.
    """

    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, out_dim, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim), nn.GELU(),
        )

    def forward(self, x):
        # x: (B, 1, F, T)
        return self.net(x)   # (B, out_dim, F/8, T/8)


# ---------------------------------------------------------------------------
# Full H-ATGNN
# ---------------------------------------------------------------------------

class HATGNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        d  = cfg.label_dim
        pd = cfg.patch_dim

        # --- Backbone ---
        self.backbone = CNNBackbone(out_dim=pd)
        self.pos_emb  = nn.Parameter(torch.randn(cfg.max_nodes, pd) * 0.02)

        # --- PGN blocks ---
        self.pgn_blocks = nn.ModuleList([
            PGNBlock(pd, k=cfg.k, dilation=i+1)
            for i in range(cfg.n_pgn)
        ])

        # --- Label embeddings (initialised from text LM if cfg.text_init) ---
        self.mood_emb    = nn.Embedding(cfg.n_mood,     d)
        self.genre_emb   = nn.Embedding(cfg.n_genre,    d)
        self.subgenre_emb = nn.Embedding(cfg.n_subgenre, d)

        # --- Cross-modal fusion ---
        self.fusion = GatedFusion(pd, cfg.clap_dim, pd)

        # --- Patch->label projection (align dims if needed) ---
        self.patch_proj = nn.Linear(pd, d) if pd != d else nn.Identity()

        # --- Hierarchical PLG ---
        self.hplg = HierarchicalPLGBlock(
            patch_dim=d,
            label_dim=d,
            n_mood=cfg.n_mood,
            n_genre=cfg.n_genre,
            n_subgenre=cfg.n_subgenre,
            k_plg=cfg.k_plg,
        )

        # --- Hierarchical LLG ---
        n_total = cfg.n_mood + cfg.n_genre + cfg.n_subgenre
        self.hllg = HierarchicalLLGBlock(n_total, d,
                                          cfg.hierarchy_mask)

        # --- Prediction heads ---
        # Patch logits
        self.patch_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # over node dim
            nn.Flatten(),
            nn.Linear(pd, cfg.n_mood + cfg.n_genre + cfg.n_subgenre),
        )

        # Label logits: per-label dot-product readout
        n_all = cfg.n_mood + cfg.n_genre + cfg.n_subgenre
        self.W_out = nn.Parameter(torch.randn(n_all, d) * 0.02)

    def initialise_from_text_embeddings(self, mood_vecs, genre_vecs, sub_vecs):
        """
        mood_vecs   : (n_mood, text_dim)  numpy or tensor
        genre_vecs  : (n_genre, text_dim)
        sub_vecs    : (n_subgenre, text_dim)
        Projects to label_dim and sets embedding weights.
        """
        d = self.cfg.label_dim

        def _proj(vecs):
            vecs = torch.as_tensor(vecs, dtype=torch.float)
            if vecs.shape[-1] != d:
                proj = nn.Linear(vecs.shape[-1], d, bias=False)
                nn.init.xavier_uniform_(proj.weight)
                vecs = proj(vecs).detach()
            return vecs

        with torch.no_grad():
            self.mood_emb.weight.copy_(_proj(mood_vecs))
            self.genre_emb.weight.copy_(_proj(genre_vecs))
            self.subgenre_emb.weight.copy_(_proj(sub_vecs))

    def forward(self, spec, clap_emb=None):
        """
        spec     : (B, 1, F, T)   log-mel spectrogram
        clap_emb : (B, clap_dim)  optional CLAP audio embedding
        """
        B = spec.size(0)

        # --- CNN backbone -> patch nodes ---
        feat = self.backbone(spec)                       # (B, pd, H, W)
        H, W = feat.shape[2], feat.shape[3]
        N = H * W
        feat = feat.flatten(2).permute(0, 2, 1)          # (B, N, pd)
        feat = feat + self.pos_emb[:N].unsqueeze(0)

        # --- PGN blocks (per-sample graph) ---
        pgn_out = []
        for b in range(B):
            x = feat[b]                                  # (N, pd)
            for pgn in self.pgn_blocks:
                x = pgn(x)
            pgn_out.append(x)
        patch_feats = torch.stack(pgn_out, dim=0)        # (B, N, pd)

        # --- Global patch representation for heads ---
        patch_global = patch_feats.mean(dim=1)           # (B, pd)

        # --- CLAP fusion ---
        if clap_emb is not None:
            patch_global = self.fusion(patch_global, clap_emb)

        # --- Patch logits ---
        y_patch = self.patch_head(patch_feats.permute(0, 2, 1))  # (B, n_labels)

        # --- Label embeddings ---
        mood_idx = torch.arange(self.cfg.n_mood,    device=spec.device)
        gen_idx  = torch.arange(self.cfg.n_genre,   device=spec.device)
        sub_idx  = torch.arange(self.cfg.n_subgenre,device=spec.device)

        # Per-sample label refinement
        label_logits = []
        for b in range(B):
            pf = self.patch_proj(patch_feats[b])         # (N, d)
            m  = self.mood_emb(mood_idx)                 # (n_mood, d)
            g  = self.genre_emb(gen_idx)                 # (n_genre, d)
            s  = self.subgenre_emb(sub_idx)              # (n_sub, d)

            # H-PLG
            m, g, s = self.hplg(pf, m, g, s)

            # H-LLG
            L = torch.cat([m, g, s], dim=0)              # (n_all, d)
            L = self.hllg(L)

            # Readout: dot each label emb with its learned output vector
            logit = (L * self.W_out).sum(-1)             # (n_all,)
            label_logits.append(logit)

        y_label = torch.stack(label_logits, dim=0)       # (B, n_all)

        y = torch.sigmoid(y_patch + y_label)             # (B, n_all)
        return y


# ---------------------------------------------------------------------------
# Hierarchical loss
# ---------------------------------------------------------------------------

class HierarchicalLoss(nn.Module):
    """
    BCE per level + parent-child consistency penalty.

    consistency penalty: if child is predicted 1, parent should also be 1.
    soft version: penalty = mean( max(0, child_prob - parent_prob) )
    """

    def __init__(self, n_mood, n_genre, n_subgenre,
                 genre_to_mood,       # (n_genre,)  index of mood parent
                 subgenre_to_genre,   # (n_subgenre,) index of genre parent
                 lam=0.5):
        super().__init__()
        self.n_mood = n_mood
        self.n_genre = n_genre
        self.n_sub = n_subgenre
        self.lam = lam
        self.register_buffer('g2m', torch.tensor(genre_to_mood, dtype=torch.long))
        self.register_buffer('s2g', torch.tensor(subgenre_to_genre, dtype=torch.long))

    def forward(self, preds, targets):
        """
        preds   : (B, n_mood + n_genre + n_sub)  sigmoid outputs
        targets : same shape, binary
        """
        nm, ng, ns = self.n_mood, self.n_genre, self.n_sub

        p_mood   = preds[:, :nm]
        p_genre  = preds[:, nm:nm+ng]
        p_sub    = preds[:, nm+ng:]

        t_mood   = targets[:, :nm].float()
        t_genre  = targets[:, nm:nm+ng].float()
        t_sub    = targets[:, nm+ng:].float()

        # Per-level BCE
        bce = (F.binary_cross_entropy(p_mood,  t_mood)
             + F.binary_cross_entropy(p_genre, t_genre)
             + F.binary_cross_entropy(p_sub,   t_sub))

        # Consistency: genre child -> mood parent
        parent_genre_mood = p_mood[:, self.g2m]   # (B, n_genre)
        cons1 = F.relu(p_genre - parent_genre_mood).mean()

        # Consistency: sub-genre -> genre parent
        parent_sub_genre = p_genre[:, self.s2g]   # (B, n_sub)
        cons2 = F.relu(p_sub - parent_sub_genre).mean()

        return bce + self.lam * (cons1 + cons2)
