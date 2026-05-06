"""PyTorch Dataset with spectral augmentation for Raman spectra."""
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d as sp_interp1d


def compute_derivative_channels(spectrum, window=11, polyorder=3):
    """
    Build 3-channel representation: [raw, d1_standardized, d2_standardized].

    Derivative channels are mathematically baseline-invariant:
        d/dx[f + a] = d/dx[f]            (1st deriv removes constant offset)
        d²/dx²[f + ax + b] = d²/dx²[f]  (2nd deriv removes linear baseline)

    Per-sample standardisation (zero mean, unit std) is applied to derivative
    channels so their magnitudes are comparable across spectra regardless of
    peak height or number of peaks.

    Args:
        spectrum : (N,) float array — preprocessed, min-max normalised
        window, polyorder : SG filter parameters; must match preprocessing.py

    Returns:
        (3, N) float32 array
    """
    s = spectrum.astype(np.float64)
    d1 = savgol_filter(s, window_length=window, polyorder=polyorder, deriv=1)
    d2 = savgol_filter(s, window_length=window, polyorder=polyorder, deriv=2)

    def _std(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    return np.stack(
        [s.astype(np.float32),
         _std(d1).astype(np.float32),
         _std(d2).astype(np.float32)],
        axis=0,
    )  # (3, N)


class RamanDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def _augment(self, spectrum: np.ndarray) -> np.ndarray:
        """Simulate realistic measurement variations to improve cross-instrument generalisation."""
        s = spectrum.copy()

        # Gaussian noise — detector shot noise
        if np.random.random() < 0.8:
            sigma = np.random.uniform(0.004, 0.030)
            s += np.random.normal(0.0, sigma, len(s))

        # Intensity scaling — laser power / sample concentration variation
        if np.random.random() < 0.6:
            s *= np.random.uniform(0.75, 1.25)

        # Residual fluorescence background (polynomial)
        if np.random.random() < 0.7:
            n = len(s)
            x = np.linspace(0, 1, n)
            a = np.random.uniform(-0.12, 0.12)
            b = np.random.uniform(-0.08, 0.08)
            c = np.random.uniform(-0.05, 0.05)
            s += a * x + b * x ** 2 + c * x ** 3

        # Wavenumber shift — expanded to ±15 pts (±50 cm⁻¹) to cover real instrument
        # calibration differences between spectrometers (was ±6 pts / ±20 cm⁻¹)
        if np.random.random() < 0.5:
            shift = np.random.randint(-15, 16)
            s = np.roll(s, shift)
            if shift > 0:
                s[:shift] = 0.0
            elif shift < 0:
                s[shift:] = 0.0

        # Random spectral dilation/compression (±0.5 %) — grating variation
        if np.random.random() < 0.3:
            factor = np.random.uniform(0.995, 1.005)
            idxs = np.arange(len(s))
            new_idxs = np.clip(idxs * factor, 0, len(s) - 1)
            s = np.interp(idxs, new_idxs, s)

        # Smooth multiplicative envelope — simulates non-uniform CCD/grating
        # quantum efficiency profile that varies across instruments
        if np.random.random() < 0.5:
            ctrl_x = np.linspace(0, 1, 4)
            ctrl_y = np.random.uniform(0.80, 1.20, 4)
            x_full = np.linspace(0, 1, len(s))
            envelope = sp_interp1d(ctrl_x, ctrl_y, kind='cubic')(x_full)
            s = s * envelope

        # Gaussian spectral broadening — simulates lower-resolution instruments
        # (larger slit width); σ 0.5–2.5 pts ≈ FWHM 1.7–8.3 cm⁻¹
        if np.random.random() < 0.4:
            sigma = np.random.uniform(0.5, 2.5)
            s = gaussian_filter1d(s, sigma=sigma)

        # Re-normalise
        s = np.clip(s, 0.0, None)
        peak = s.max()
        if peak > 0:
            s /= peak

        return s.astype(np.float32)

    def __getitem__(self, idx):
        spectrum = self.X[idx]
        if self.augment:
            spectrum = self._augment(spectrum)   # augment 1D first
        channels = compute_derivative_channels(spectrum)  # then build (3, 1024)
        return torch.tensor(channels), torch.tensor(self.y[idx])
