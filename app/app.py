"""
Raman Mineral Fingerprinting — Streamlit Web Application
Department of Chemical Engineering

Upload a two-column Raman spectrum file (wavenumber, intensity) to:
  1. Identify the mineral with confidence scores
  2. Decompose mixtures into percentage compositions
  3. Visualise which spectral peaks drove the model decision (Grad-CAM)
"""
import io
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

# ---- path setup ----------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import RamanPredictor
from config import STANDARD_GRID

# ---- page config ---------------------------------------------------------
st.set_page_config(
    page_title="Raman Mineral Fingerprinting",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- CSS -----------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #f0f4ff; border-radius: 8px; padding: 1rem;
        text-align: center; margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- helpers -------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def load_predictor():
    return RamanPredictor(
        model_dir=str(ROOT / "models"),
        data_dir=str(ROOT / "data"),
    )


@st.cache_data(show_spinner=False)
def load_training_info():
    p = ROOT / "models" / "training_info.json"
    if p.exists():
        with open(p) as fh:
            return json.load(fh)
    return {}


def parse_spectrum(text: str):
    """
    Parse a two-column text/CSV spectrum.
    Lines starting with '#' are treated as comments.
    Returns (wavenumbers, intensities) or (None, None) on failure.
    """
    rows_wn, rows_in = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = line.replace(";", ",").replace("\t", ",").split(",")
            if len(parts) >= 2:
                wn = float(parts[0])
                inten = float(parts[1])
                rows_wn.append(wn)
                rows_in.append(inten)
        except ValueError:
            continue
    if len(rows_wn) < 20:
        return None, None
    return np.array(rows_wn), np.array(rows_in)


def confidence_color(conf: float) -> str:
    if conf >= 0.75:
        return "#2ecc71"
    if conf >= 0.50:
        return "#f39c12"
    return "#e74c3c"


# ---- sidebar -------------------------------------------------------------

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Raman_spectroscopy.svg/320px-Raman_spectroscopy.svg.png",
             use_container_width=True, caption="Raman Spectroscopy")
    st.header("Settings")

    mixture_threshold = st.slider(
        "Mixture detection threshold",
        min_value=0.40, max_value=0.95, value=0.70, step=0.05,
        help=(
            "If the CNN's top-class confidence falls below this value the system "
            "automatically switches to NNLS mixture deconvolution."
        ),
    )
    min_fraction = st.slider(
        "Min mixture fraction (%)",
        min_value=1, max_value=20, value=5, step=1,
        help="Components below this fraction are suppressed in mixture output.",
    ) / 100.0

    st.divider()
    st.subheader("Model Stats")
    info = load_training_info()
    if info:
        st.metric("Mineral classes", info.get("n_classes", "—"))
        st.metric("Test accuracy", f"{info.get('test_accuracy', 0)*100:.1f}%")
        st.metric("Training spectra", info.get("n_train", "—"))
    else:
        st.info("Train the model first (run `python run.py`).")

    st.divider()
    st.caption("Data: RRUFF Project (rruff.info)")
    st.caption("Dept. of Chemical Engineering")

# ---- main panel ----------------------------------------------------------

st.title("🔬 Raman Mineral Fingerprinting System")
st.markdown(
    "*1D CNN + Attention · NNLS Mixture Deconvolution · Grad-CAM Explainability*"
)
st.divider()

# Input section
col_input, col_results = st.columns([1, 2], gap="large")

with col_input:
    st.subheader("Input Spectrum")
    mode = st.radio("Source", ["Upload file", "Paste data"], horizontal=True)

    wn, inten = None, None

    if mode == "Upload file":
        uploaded = st.file_uploader(
            "Upload spectrum (.txt, .csv, .dat)",
            type=["txt", "csv", "dat"],
            help="Two-column file: wavenumber (cm⁻¹) and intensity.",
        )
        if uploaded:
            text = uploaded.read().decode("utf-8", errors="ignore")
            wn, inten = parse_spectrum(text)
            if wn is None:
                st.error("Could not parse file. Ensure two numeric columns separated by commas or tabs.")

    else:
        sample = "100.0, 200\n150.0, 310\n200.0, 290\n..."
        pasted = st.text_area(
            "Paste spectrum data (wavenumber, intensity per row):",
            height=220,
            placeholder=sample,
        )
        if pasted.strip():
            wn, inten = parse_spectrum(pasted)
            if wn is None:
                st.error("Could not parse pasted data.")

    if wn is not None and inten is not None:
        st.success(f"Loaded {len(wn)} data points  |  {wn.min():.0f}–{wn.max():.0f} cm⁻¹")
        fig_raw = go.Figure(
            go.Scatter(x=wn, y=inten, mode="lines",
                       line=dict(color="#2980b9", width=1.5), name="Raw")
        )
        fig_raw.update_layout(
            title="Raw Input Spectrum",
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Intensity (a.u.)",
            height=250, margin=dict(t=35, b=35, l=50, r=10),
            plot_bgcolor="#fafafa",
        )
        st.plotly_chart(fig_raw, use_container_width=True)

    analyze = st.button(
        "Analyze Spectrum",
        type="primary",
        use_container_width=True,
        disabled=(wn is None),
    )

# Results section
with col_results:
    st.subheader("Analysis Results")

    if wn is not None and analyze:
        with st.spinner("Preprocessing and running model…"):
            try:
                predictor = load_predictor()
                result = predictor.predict(
                    wn, inten,
                    mixture_threshold=mixture_threshold,
                    min_fraction=min_fraction,
                )
            except FileNotFoundError:
                st.error(
                    "Model not found. Run `python run.py` to download data and train the model first."
                )
                st.stop()

        if "error" in result:
            st.error(result["error"])
            st.stop()

        # ---- Store in session ----------------------------------------
        st.session_state["result"] = result

    result = st.session_state.get("result")
    if result is None:
        st.info("Upload a spectrum and click **Analyze Spectrum** to see results.")
        st.stop()

    tab_id, tab_mix, tab_cam, tab_spec = st.tabs(
        ["Identification", "Mixture Analysis", "Grad-CAM", "Preprocessed Spectrum"]
    )

    # ------------------------------------------------------------------
    # Tab 1: Identification
    # ------------------------------------------------------------------
    with tab_id:
        top = result["top_predictions"]
        conf = result["top_confidence"]
        best_name, best_conf = top[0]

        if not result["is_mixture"]:
            st.success(f"**{best_name}** — {best_conf*100:.1f}% confidence")
        else:
            st.warning(
                f"Low confidence ({best_conf*100:.1f}%) — likely a **mixture**. "
                "See the Mixture Analysis tab."
            )

        minerals = [p[0] for p in top]
        confs = [p[1] * 100 for p in top]
        colors = [confidence_color(p[1]) for p in top]

        fig_bar = go.Figure(
            go.Bar(
                x=confs[::-1], y=minerals[::-1],
                orientation="h",
                marker_color=colors[::-1],
                text=[f"{c:.1f}%" for c in confs[::-1]],
                textposition="inside",
            )
        )
        fig_bar.update_layout(
            title="Top-5 CNN Predictions",
            xaxis=dict(title="Confidence (%)", range=[0, 100]),
            height=280,
            margin=dict(t=40, b=30, l=140, r=20),
            plot_bgcolor="#fafafa",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ------------------------------------------------------------------
    # Tab 2: Mixture Analysis
    # ------------------------------------------------------------------
    with tab_mix:
        if result["is_mixture"] and result["mixture_components"]:
            components = result["mixture_components"]
            labels = [c[0] for c in components]
            fracs = [c[1] * 100 for c in components]

            c1, c2 = st.columns(2)
            with c1:
                fig_pie = go.Figure(
                    go.Pie(
                        labels=labels, values=fracs,
                        hole=0.35, textinfo="label+percent",
                    )
                )
                fig_pie.update_layout(
                    title="Mixture Composition (NNLS)",
                    height=320, margin=dict(t=40),
                    showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with c2:
                df_comp = pd.DataFrame({
                    "Mineral": labels,
                    "Fraction (%)": [f"{v:.1f}" for v in fracs],
                })
                st.table(df_comp)
                st.caption(
                    "Non-Negative Least Squares (NNLS) decomposition: "
                    "observed spectrum ≈ Σ aᵢ·Sᵢ (aᵢ ≥ 0)."
                )

        elif result["is_mixture"]:
            st.info("NNLS did not find significant components. Try lowering the mixture threshold.")
        else:
            st.info(
                f"The model is confident this is a **pure** sample "
                f"({result['top_confidence']*100:.1f}% ≥ {mixture_threshold*100:.0f}% threshold). "
                "Lower the threshold to force mixture analysis."
            )

    # ------------------------------------------------------------------
    # Tab 3: Grad-CAM
    # ------------------------------------------------------------------
    with tab_cam:
        spectrum = result["spectrum"]
        cam = result.get("gradcam")
        best_name = result["top_predictions"][0][0]

        if cam is not None:
            fig_cam = go.Figure()

            # Highlight important regions
            step = 8
            for i in range(0, len(STANDARD_GRID) - step, step):
                alpha = float(cam[i]) * 0.45
                if alpha > 0.01:
                    fig_cam.add_vrect(
                        x0=STANDARD_GRID[i],
                        x1=STANDARD_GRID[i + step],
                        fillcolor=f"rgba(231,76,60,{alpha:.3f})",
                        line_width=0,
                        layer="below",
                    )

            fig_cam.add_trace(
                go.Scatter(
                    x=STANDARD_GRID, y=spectrum,
                    mode="lines",
                    line=dict(color="#2c3e50", width=2),
                    name="Preprocessed spectrum",
                )
            )
            fig_cam.update_layout(
                title=f"Grad-CAM Attribution — predicted: {best_name}",
                xaxis_title="Wavenumber (cm⁻¹)",
                yaxis_title="Normalised Intensity",
                height=360,
                margin=dict(t=45, b=40, l=60, r=20),
                plot_bgcolor="#fafafa",
            )
            st.plotly_chart(fig_cam, use_container_width=True)
            st.caption(
                "Red shading indicates spectral regions with **high gradient-weighted activation** — "
                "the model's most diagnostic peaks for its prediction."
            )
        else:
            st.info("Grad-CAM could not be computed for this spectrum.")

    # ------------------------------------------------------------------
    # Tab 4: Preprocessed spectrum
    # ------------------------------------------------------------------
    with tab_spec:
        spectrum = result["spectrum"]
        fig_pre = go.Figure(
            go.Scatter(
                x=STANDARD_GRID, y=spectrum,
                mode="lines",
                line=dict(color="#8e44ad", width=1.5),
                name="Preprocessed",
            )
        )
        fig_pre.update_layout(
            title="Preprocessed Spectrum (ALS baseline-corrected, SG-smoothed, normalised)",
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Normalised Intensity",
            height=320,
            margin=dict(t=45, b=40),
            plot_bgcolor="#fafafa",
        )
        st.plotly_chart(fig_pre, use_container_width=True)
        st.caption("1024-point uniform grid · 100–3500 cm⁻¹")
