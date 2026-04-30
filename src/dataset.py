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
        """Simulate realistic measurement variations to prevent overfitting."""
        s = spectrum.copy()

        # Gaussian noise — models detector shot noise
        if np.random.random() < 0.8:
            sigma = np.random.uniform(0.004, 0.025)
            s += np.random.normal(0.0, sigma, len(s))

        # Intensity scaling — models laser power / sample concentration variation
        if np.random.random() < 0.6:
            s *= np.random.uniform(0.80, 1.20)

        # Smooth polynomial baseline drift — residual fluorescence
        if np.random.random() < 0.5:
            n = len(s)
            x = np.linspace(0, 1, n)
            a = np.random.uniform(-0.06, 0.06)
            b = np.random.uniform(-0.04, 0.04)
            s += a * x + b * x ** 2

        # Small wavenumber shift (±3 points) — instrument calibration drift
        if np.random.random() < 0.3:
            shift = np.random.randint(-3, 4)
            s = np.roll(s, shift)
            if shift > 0:
                s[:shift] = 0.0
            elif shift < 0:
                s[shift:] = 0.0

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
