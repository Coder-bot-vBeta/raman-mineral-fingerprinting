WAVENUMBER_MIN = 100    # cm⁻¹
WAVENUMBER_MAX = 3500   # cm⁻¹
N_POINTS = 1024         # interpolation grid size
MIXTURE_THRESHOLD = 0.70  # CNN confidence below this → NNLS unmixing
MIN_SAMPLES_PER_CLASS = 10  # discard minerals with fewer spectra

import numpy as np
STANDARD_GRID = np.linspace(WAVENUMBER_MIN, WAVENUMBER_MAX, N_POINTS)
