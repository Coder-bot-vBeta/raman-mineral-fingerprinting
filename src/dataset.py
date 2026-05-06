"""PyTorch Dataset with spectral augmentation for Raman spectra."""
import numpy as np
import torch
from torch.utils.data import Dataset


class RamanDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def _augment(self, spectrum: np.ndarray) -> np.ndarray:
        """Simulate realistic measurement variations to improve generalisation."""
        s = spectrum.copy()

        # Gaussian noise — detector shot noise
        if np.random.random() < 0.8:
            sigma = np.random.uniform(0.004, 0.030)
            s += np.random.normal(0.0, sigma, len(s))

        # Intensity scaling — laser power / sample concentration variation
        if np.random.random() < 0.6:
            s *= np.random.uniform(0.75, 1.25)

        # Residual fluorescence background (polynomial) — most important for
        # cross-instrument robustness; increased amplitude vs. original
        if np.random.random() < 0.7:
            n = len(s)
            x = np.linspace(0, 1, n)
            a = np.random.uniform(-0.12, 0.12)
            b = np.random.uniform(-0.08, 0.08)
            c = np.random.uniform(-0.05, 0.05)
            s += a * x + b * x ** 2 + c * x ** 3

        # Wavenumber shift ±6 points — broader calibration drift envelope
        if np.random.random() < 0.4:
            shift = np.random.randint(-6, 7)
            s = np.roll(s, shift)
            if shift > 0:
                s[:shift] = 0.0
            elif shift < 0:
                s[shift:] = 0.0

        # Random spectral dilation/compression (±0.5 %) — grating variation
        if np.random.random() < 0.3:
            factor = np.random.uniform(0.995, 1.005)
            idxs = np.arange(len(s))
            new_idxs = idxs * factor
            new_idxs = np.clip(new_idxs, 0, len(s) - 1)
            s = np.interp(idxs, new_idxs, s)

        # Re-normalise
        s = np.clip(s, 0.0, None)
        peak = s.max()
        if peak > 0:
            s /= peak

        return s.astype(np.float32)

    def __getitem__(self, idx):
        spectrum = self.X[idx]
        if self.augment:
            spectrum = self._augment(spectrum)
        return torch.tensor(spectrum), torch.tensor(self.y[idx])
