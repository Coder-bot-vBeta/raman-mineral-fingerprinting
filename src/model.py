"""
1D ResNet with self-attention for Raman spectrum classification.

Architecture:
    3-channel input: [raw spectrum, 1st derivative, 2nd derivative]
    4 × ResBlock1D (two Conv1D + residual skip + MaxPool)
    → Multi-head self-attention (4 heads)
    → Global Average Pooling
    → Dropout → FC(512) → FC(n_classes)

The derivative channels are mathematically baseline-free:
    d/dx[f + a] = d/dx[f]        (1st derivative removes constant offset)
    d²/dx²[f + ax + b] = d²/dx²[f]  (2nd derivative removes linear baseline)

This makes the model invariant to residual fluorescence baselines that vary
across instruments, which is the primary cause of cross-instrument failures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    """
    Residual block with two Conv1d layers and a skip connection.
    Skip uses 1×1 conv when in_ch != out_ch. Pool (MaxPool1d) applied after
    residual addition so the skip path sees full spatial resolution.
    """

    def __init__(self, in_ch, out_ch, kernel=7, pool=None):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool1d(pool) if pool else None
        self.skip  = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        ) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        if self.pool is not None:
            out = self.pool(out)
        return out


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


class RamanResNet(nn.Module):
    """
    Raman spectrum classifier with residual blocks.

    Input : (B, 3, 1024) — [raw, d1, d2] channels
            Also accepts (B, 1, 1024) and (B, 1024) for backwards compatibility
            (single-channel input is repeated across 3 channels).
    Output: (B, n_classes) — raw logits
    """

    def __init__(self, n_classes: int, in_channels: int = 3):
        super().__init__()
        # Spatial progression: 1024 → 256 → 64 → 16 → 8  (pool 4,4,4,2)
        # Channel progression:   in  →  64 → 128 → 256 → 256
        self.block1    = ResBlock1D(in_channels, 64,  kernel=11, pool=4)
        self.block2    = ResBlock1D(64,          128, kernel=7,  pool=4)
        self.block3    = ResBlock1D(128,         256, kernel=5,  pool=4)
        self.block4    = ResBlock1D(256,         256, kernel=3,  pool=2)
        self.attention = SelfAttention1D(256, n_heads=4)
        self.gap       = nn.AdaptiveAvgPool1d(1)
        self.dropout   = nn.Dropout(0.5)
        self.fc1       = nn.Linear(256, 512)
        self.fc2       = nn.Linear(512, n_classes)

    def _ensure_3ch(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (B, 1, L)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1)      # duplicate to 3 channels
        return x

    def _extract_features(self, x):
        x = self._ensure_3ch(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.attention(x)
        return x  # (B, 256, 8)

    def forward(self, x):
        x = self._extract_features(x)
        x = self.gap(x).squeeze(-1)    # (B, 256)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    # ------------------------------------------------------------------
    # Grad-CAM support
    # ------------------------------------------------------------------
    def get_cam_activations_and_logits(self, x):
        """Return (feature_map, logits) with gradient graph intact."""
        x = self._ensure_3ch(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        feat = self.block4(x)           # (B, 256, 8)
        feat.retain_grad()              # non-leaf node needs explicit retention
        attn_out = self.attention(feat)
        pooled   = self.gap(attn_out).squeeze(-1)
        pooled   = self.dropout(pooled)
        out      = F.relu(self.fc1(pooled))
        out      = self.dropout(out)
        logits   = self.fc2(out)
        return feat, logits


class TemperatureScaler(nn.Module):
    """
    Post-hoc confidence calibration via temperature scaling (Guo et al. 2017).
    Does not change the argmax — only softens/sharpens probabilities.
    T > 1 → softer (more uncertain); T < 1 → sharper (more confident).
    """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([float(temperature)]))

    def forward(self, logits):
        return logits / self.temperature.clamp(min=0.05)

    def calibrate(self, logits_val, labels_val, lr=0.01, max_iter=50):
        """
        Find optimal T by minimising NLL on pre-collected validation logits.
        Uses LBFGS (guaranteed to converge for this 1D convex problem).

        Args:
            logits_val : (N_val, n_classes) tensor — raw model logits, no grad needed
            labels_val : (N_val,) tensor — integer class labels

        Returns:
            float: calibrated temperature value
        """
        self.train()
        nll = nn.CrossEntropyLoss()
        opt = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = nll(self.forward(logits_val), labels_val)
            loss.backward()
            return loss

        opt.step(closure)
        t = float(self.temperature.item())
        print(f"  Calibrated temperature T = {t:.4f}")
        return t
