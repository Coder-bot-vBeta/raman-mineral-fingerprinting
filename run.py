"""
One-command launcher for the Raman Mineral Fingerprinting system.

Usage
-----
    python run.py              # generate data + train + launch Streamlit app
    python run.py --app        # launch app only (model already trained)
    python run.py --train      # generate data + train only (no app launch)
    python run.py --rruff      # attempt RRUFF bulk download instead of synthetic data

Data strategy
-------------
By default, high-quality synthetic spectra are generated from published
Raman peak positions (RRUFF project + literature). This trains instantly
and covers 90+ mineral classes. Optionally, pass --rruff to attempt
downloading the full RRUFF database zip files (~1 GB total) for maximum
coverage — only recommended on fast internet connections.
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def data_ready():
    return (ROOT / "data" / "processed" / "X.npy").exists()


def model_ready():
    return (ROOT / "models" / "best_model.pth").exists()


def run_synthetic():
    print("\n" + "=" * 60)
    print("  STEP 1 — Generating synthetic Raman training data")
    print("           (based on published RRUFF peak positions)")
    print("=" * 60)
    from synthetic_data import generate_dataset
    generate_dataset(data_dir=str(ROOT / "data"), n_per_mineral=30)


def run_rruff_download():
    print("\n" + "=" * 60)
    print("  STEP 1 — Downloading RRUFF database (may take a while)")
    print("=" * 60)
    from data_pipeline import download_rruff, build_dataset
    download_rruff(str(ROOT / "data"))
    build_dataset(str(ROOT / "data"))


def run_training():
    print("\n" + "=" * 60)
    print("  STEP 2 — Training 1D-CNN with self-attention")
    print("=" * 60)
    from train import train
    train(
        data_dir=str(ROOT / "data"),
        model_dir=str(ROOT / "models"),
    )


def launch_app():
    print("\n" + "=" * 60)
    print("  Launching Streamlit app  →  http://localhost:8501")
    print("=" * 60)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run",
         str(ROOT / "app" / "app.py"),
         "--server.headless", "false"],
        check=True,
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    use_rruff = "--rruff" in args

    if "--app" in args:
        if not model_ready():
            print("ERROR: No trained model found. Run `python run.py` first.")
            sys.exit(1)
        launch_app()

    elif "--train" in args:
        if not data_ready():
            if use_rruff:
                run_rruff_download()
            else:
                run_synthetic()
        run_training()

    else:
        # Full pipeline
        if not data_ready():
            if use_rruff:
                run_rruff_download()
            else:
                run_synthetic()
        else:
            print("Dataset already exists — skipping data generation.")

        if not model_ready():
            run_training()
        else:
            print("Trained model already exists — skipping training.")
            print("  (Delete models/best_model.pth to force retrain.)")

        launch_app()
