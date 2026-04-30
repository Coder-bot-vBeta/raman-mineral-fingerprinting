"""
1D CNN with self-attention for Raman spectrum classification.

Architecture:
    4 × ConvBlock (Conv1D + BatchNorm + ReLU + MaxPool)
    → Multi-head self-attention (4 heads)
    → Global Average Pooling
    → Dropout → FC(512) → FC(n_classes)

The self-attention lets the model weight diagnostic peaks over background regions
without being position-bound, which is important because Raman shifts can vary
slightly with laser wavelength and instrument calibration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, pool=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention1D(nn.Module):
    """Multi-head self-attention over the spatial (wavenumber) dimension."""

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, C, L)  →  transpose for attention  →  (B, L, C)
        x_t = x.permute(0, 2, 1)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        out = self.norm(x_t + attn_out)
        return out.permute(0, 2, 1)  # back to (B, C, L)


class RamanNet(nn.Module):
    """
    Raman spectrum classifier.

    Input : (B, 1024) or (B, 1, 1024)
    Output: (B, n_classes) — raw logits
    """

    def __init__(self, n_classes: int, input_length: int = 1024):
        super().__init__()
        # Convolutional feature extractor
        # 1024 → 256 → 64 → 16 → 8  (via pool sizes 4,4,4,2)
        self.block1 = ConvBlock(1, 32, kernel=11, pool=4)
        self.block2 = ConvBlock(32, 64, kernel=7, pool=4)
        self.block3 = ConvBlock(64, 128, kernel=5, pool=4)
        self.block4 = ConvBlock(128, 256, kernel=3, pool=2)

        # Self-attention over 8 spatial positions, 256 channels
        self.attention = SelfAttention1D(256, n_heads=4)

        # Classifier head
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, n_classes)

    def _extract_features(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,L)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.attention(x)
        return x  # (B, 256, 8)

    def forward(self, x):
        x = self._extract_features(x)
        x = self.gap(x).squeeze(-1)   # (B, 256)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    # ------------------------------------------------------------------
    # Grad-CAM support — expose last-conv activations + allow gradients
    # ------------------------------------------------------------------
    def get_cam_activations_and_logits(self, x):
        """Return (feature_map, logits) with gradient graph intact."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        feat = self.block4(x)          # (B, 256, 8) — kept for grad
        attn_out = self.attention(feat)
        pooled = self.gap(attn_out).squeeze(-1)
        pooled = self.dropout(pooled)
        out = F.relu(self.fc1(pooled))
        out = self.dropout(out)
        logits = self.fc2(out)
        return feat, logits
