"""
Test real NASA Ames Ramdb spectra through RamanPredictor and save result panels.

Source: NASA Ames Raman Spectroscopic Database (Ramdb v1.00)
        https://www.astrochemistry.org/ramdb/
        Laser: 532 nm, T=293 K  — used for external validation only.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from predict import RamanPredictor
from config import STANDARD_GRID

NASA_FILES = {
    "Olivine": r"D:\Mini Project\testdata\ramdb-v1.00-Olivine_532_microimaging_homogeneous_293K_none_0_solid_crystals_09082022_jnt9lc2nlcqg8HBoHlz.csv",
    "Calcite (CaCO3)": r"D:\Mini Project\testdata\ramdb-v1.00-CalciumCarbonate_532_microimaging_homogeneous_293K_none_0_solid_crystals_09082022_crfnhahuo5d4eHiunFa.csv",
}

EXPECTED = {
    "Olivine": "Olivine",
    "Calcite (CaCO3)": "Calcite",
}


def parse_nasa_csv(path):
    wn, inten = [], []
    with open(path) as f:
        in_data = False
        for line in f:
            line = line.strip()
            if line.startswith("XYDATA"):
                in_data = True
                continue
            if in_data and "," in line:
                try:
                    w, i = line.split(",")
                    wn.append(float(w))
                    inten.append(float(i))
                except ValueError:
                    pass
    return np.array(wn), np.array(inten)


def conf_color(c):
    if c >= 0.75: return "#2ecc71"
    if c >= 0.50: return "#f39c12"
    return "#e74c3c"


def save_result_figure(mineral_name, wn_raw, inten_raw, result, out_path):
    top   = result["top_predictions"]
    conf  = result["top_confidence"]
    is_mix = result["is_mixture"]
    proc  = result["spectrum"]
    cam   = result["gradcam"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("white")

    pred_label  = top[0][0]
    exp_label   = EXPECTED[mineral_name]
    match_icon  = "PASS" if exp_label.lower() in pred_label.lower() else "FAIL"

    fig.suptitle(
        f"NASA Ramdb External Test  —  {mineral_name}\n"
        f"Predicted: {pred_label}  |  Confidence: {conf*100:.1f}%  |  "
        f"Mode: {'Mixture' if is_mix else 'Pure sample'}  |  [{match_icon}]",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── Panel 1: Raw spectrum ────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(wn_raw, inten_raw, color="#2980b9", lw=0.8)
    ax.set_title("Raw Input Spectrum (NASA Ramdb, 532 nm)", fontsize=10)
    ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Intensity (counts)")
    ax.set_facecolor("#fafafa"); ax.grid(alpha=0.3)

    # ── Panel 2: Top-5 bar chart ─────────────────────────────────────────
    ax = axes[0, 1]
    names  = [p[0] for p in top]
    values = [p[1] * 100 for p in top]
    colors = [conf_color(p[1]) for p in top]
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1])
    for bar, val in zip(bars, values[::-1]):
        ax.text(min(val + 1, 98), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_title("Top-5 CNN Predictions", fontsize=10)
    ax.set_xlabel("Confidence (%)"); ax.set_facecolor("#fafafa"); ax.grid(alpha=0.3, axis="x")

    # ── Panel 3: Grad-CAM overlay ────────────────────────────────────────
    ax = axes[1, 0]
    if cam is not None:
        step = 8
        for i in range(0, len(STANDARD_GRID) - step, step):
            alpha = float(cam[i]) * 0.5
            if alpha > 0.01:
                ax.axvspan(STANDARD_GRID[i], STANDARD_GRID[i + step],
                           alpha=alpha, color="#e74c3c", linewidth=0)
    ax.plot(STANDARD_GRID, proc, color="#2c3e50", lw=1.5)
    ax.set_title("Grad-CAM Attribution  (red = diagnostic regions)", fontsize=10)
    ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Normalised Intensity")
    ax.set_facecolor("#fafafa"); ax.grid(alpha=0.3)

    # ── Panel 4: Preprocessed spectrum ───────────────────────────────────
    ax = axes[1, 1]
    ax.plot(STANDARD_GRID, proc, color="#8e44ad", lw=1.5)
    ax.set_title("Preprocessed Spectrum  (ALS baseline, SG smooth, normalised)", fontsize=10)
    ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Normalised Intensity")
    ax.set_facecolor("#fafafa"); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def run():
    predictor = RamanPredictor(
        model_dir=str(ROOT / "models"),
        data_dir=str(ROOT / "data"),
    )
    out_dir = ROOT / "nasa_results"
    out_dir.mkdir(exist_ok=True)

    for mineral_name, path in NASA_FILES.items():
        print(f"\n{'='*60}\nTesting: {mineral_name}")
        wn, inten = parse_nasa_csv(path)
        print(f"  Points: {len(wn)}, range {wn.min():.0f}-{wn.max():.0f} cm-1")

        result = predictor.predict(wn, inten, mixture_threshold=0.70)
        top = result["top_predictions"]
        print(f"  Top-5 predictions:")
        for m, c in top:
            print(f"    {m}: {c*100:.1f}%")

        safe = mineral_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
        out_path = out_dir / f"{safe}_NASA_result.png"
        save_result_figure(mineral_name, wn, inten, result, out_path)

    print("\nAll done.")


if __name__ == "__main__":
    run()
