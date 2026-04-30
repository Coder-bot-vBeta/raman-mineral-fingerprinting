"""
Raman spectrum preprocessing pipeline.
Steps: cosmic-ray removal → ALS baseline correction → Savitzky-Golay smoothing
       → interpolation to standard grid → min-max normalization.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from config import STANDARD_GRID, WAVENUMBER_MIN, WAVENUMBER_MAX, N_POINTS


def remove_cosmic_rays(intensity, z_threshold=5.0):
    """Replace spike pixels (|modified Z-score| > threshold) with local median."""
    cleaned = intensity.copy()
    diff = np.diff(intensity)
    mad = np.median(np.abs(diff - np.median(diff)))
    if mad == 0:
        return cleaned
    mz = 0.6745 * np.abs(diff - np.median(diff)) / mad
    spike_mask = mz > z_threshold
    spike_indices = np.where(spike_mask)[0]
    for idx in spike_indices:
        lo = max(0, idx - 3)
        hi = min(len(intensity), idx + 4)
        neighbors = [i for i in range(lo, hi) if i != idx and i != idx + 1]
        if neighbors:
            cleaned[idx] = np.median(intensity[neighbors])
            if idx + 1 < len(intensity):
                cleaned[idx + 1] = np.median(intensity[neighbors])
    return cleaned


def baseline_als(y, lam=1e5, p=0.01, n_iter=15):
    """
    Asymmetric Least Squares baseline correction (Eilers & Boelens 2005).
    Removes fluorescence background while preserving Raman peaks.
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L), format="csc")
    D = lam * D.T.dot(D)
    w = np.ones(L)
    z = np.zeros(L)
    for _ in range(n_iter):
        W = sparse.diags(w, 0, format="csc")
        z = spsolve(W + D, w * y)
        w = p * (y > z) + (1.0 - p) * (y < z)
    return z


def preprocess_spectrum(wavenumbers, intensities):
    """
    Full preprocessing pipeline for one Raman spectrum.

    Parameters
    ----------
    wavenumbers : array-like  (cm⁻¹)
    intensities : array-like

    Returns
    -------
    np.ndarray of shape (N_POINTS,), float32, or None if the spectrum is invalid.
    """
    wn = np.asarray(wavenumbers, dtype=float)
    inten = np.asarray(intensities, dtype=float)

    # Sort and deduplicate
    order = np.argsort(wn)
    wn, inten = wn[order], inten[order]
    _, uniq = np.unique(wn, return_index=True)
    wn, inten = wn[uniq], inten[uniq]

    # Restrict to range with margin
    mask = (wn >= WAVENUMBER_MIN - 100) & (wn <= WAVENUMBER_MAX + 100)
    wn, inten = wn[mask], inten[mask]
    if len(wn) < 20:
        return None

    # Interpolate to standard grid
    f_interp = interp1d(wn, inten, kind="linear", bounds_error=False, fill_value=0.0)
    spectrum = f_interp(STANDARD_GRID)

    # Cosmic ray removal
    spectrum = remove_cosmic_rays(spectrum)

    # ALS baseline correction
    baseline = baseline_als(spectrum)
    spectrum = np.clip(spectrum - baseline, 0.0, None)

    # Savitzky-Golay smoothing (window=11, degree=3 — preserves peak widths)
    spectrum = savgol_filter(spectrum, window_length=11, polyorder=3)
    spectrum = np.clip(spectrum, 0.0, None)

    # Min-max normalization
    peak = spectrum.max()
    if peak <= 0:
        return None
    spectrum = spectrum / peak

    return spectrum.astype(np.float32)
