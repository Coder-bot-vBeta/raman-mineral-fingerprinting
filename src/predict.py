"""
Inference module: CNN classification + NNLS mixture deconvolution + Grad-CAM.

Usage
-----
    predictor = RamanPredictor()
    result = predictor.predict(wavenumbers, intensities)
    # result keys: top_predictions, is_mixture, mixture_components, gradcam, spectrum
"""
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.optimize import nnls
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent))
from model import RamanNet
from preprocessing import preprocess_spectrum
from config import STANDARD_GRID, MIXTURE_THRESHOLD


class RamanPredictor:
    def __init__(
        self,
        model_dir: str = "models",
        data_dir: str = "data",
    ):
        model_dir = Path(model_dir)
        processed_dir = Path(data_dir) / "processed"

        # Label map
        labels_df = pd.read_csv(processed_dir / "label_map.csv")
        self.label_to_mineral = dict(zip(labels_df["label"], labels_df["mineral"]))
        self.mineral_names = list(labels_df["mineral"])
        n_classes = len(labels_df)

        # Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RamanNet(n_classes).to(self.device)
        self.model.load_state_dict(
            torch.load(model_dir / "best_model.pth", map_location=self.device)
        )
        self.model.eval()

        # Reference spectra matrix for NNLS  shape: (n_classes, N_POINTS)
        self.reference_spectra = np.load(processed_dir / "reference_spectra.npy")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(
        self,
        wavenumbers,
        intensities,
        mixture_threshold: float = MIXTURE_THRESHOLD,
        top_k: int = 5,
        min_fraction: float = 0.05,
    ) -> dict:
        """
        Identify mineral(s) from a raw Raman spectrum.

        Returns
        -------
        dict with keys:
            spectrum          : preprocessed 1D numpy array (1024 pts)
            top_predictions   : list of (mineral_name, confidence) — top_k
            top_confidence    : float
            is_mixture        : bool
            mixture_components: list of (mineral_name, fraction) or None
            gradcam           : 1D numpy array (1024 pts) or None
        """
        spectrum = preprocess_spectrum(
            np.asarray(wavenumbers, dtype=float),
            np.asarray(intensities, dtype=float),
        )
        if spectrum is None:
            return {"error": "Preprocessing failed — check that the spectrum covers 100–3500 cm⁻¹."}

        # CNN forward pass
        tensor = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        top_idx = np.argsort(probs)[::-1][:top_k]
        top_predictions = [(self.label_to_mineral[i], float(probs[i])) for i in top_idx]
        top_confidence = float(probs[top_idx[0]])

        is_mixture = top_confidence < mixture_threshold
        mixture_components = None
        if is_mixture:
            mixture_components = self._nnls_unmix(spectrum, min_fraction)

        gradcam = self._compute_gradcam(spectrum, int(top_idx[0]))

        return {
            "spectrum": spectrum,
            "top_predictions": top_predictions,
            "top_confidence": top_confidence,
            "is_mixture": is_mixture,
            "mixture_components": mixture_components,
            "gradcam": gradcam,
        }

    # ------------------------------------------------------------------
    # NNLS spectral unmixing
    # ------------------------------------------------------------------

    def _nnls_unmix(self, spectrum: np.ndarray, min_fraction: float):
        """
        Solve  argmin ||spectrum - A·x||²  subject to x ≥ 0.
        A : (N_POINTS × n_classes) matrix of reference spectra (column-wise).
        """
        A = self.reference_spectra.T  # (N_POINTS, n_classes)
        coeffs, _ = nnls(A, spectrum)
        total = coeffs.sum()
        if total <= 0:
            return []
        fractions = coeffs / total

        components = [
            (self.mineral_names[i], float(fractions[i]))
            for i in np.argsort(fractions)[::-1]
            if fractions[i] >= min_fraction
        ]
        return components[:10]

    # ------------------------------------------------------------------
    # Grad-CAM (1D)
    # ------------------------------------------------------------------

    def _compute_gradcam(self, spectrum: np.ndarray, target_class: int):
        """
        Grad-CAM over the last conv block.  Returns a 1024-point saliency map
        (values in [0, 1]) where high values indicate diagnostic spectral regions.
        """
        self.model.eval()
        tensor = (
            torch.tensor(spectrum, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
            .requires_grad_(True)
        )

        try:
            feat, logits = self.model.get_cam_activations_and_logits(tensor)

            self.model.zero_grad()
            score = logits[0, target_class]
            score.backward()

            # feat: (1, 256, 8)
            gradients = feat.grad  # same shape
            if gradients is None:
                return None

            # Channel-wise global average of gradients
            weights = gradients[0].mean(dim=-1)          # (256,)
            cam = (weights.unsqueeze(-1) * feat[0]).sum(0)  # (8,)
            cam = torch.relu(cam).detach().cpu().numpy()

            # Upsample to 1024 points
            x_orig = np.linspace(0, 1, len(cam))
            x_new = np.linspace(0, 1, len(STANDARD_GRID))
            cam_up = interp1d(x_orig, cam, kind="cubic")(x_new)
            cam_up = np.clip(cam_up, 0, None)
            if cam_up.max() > 0:
                cam_up /= cam_up.max()
            return cam_up.astype(np.float32)

        except Exception:
            return None
