"""
Downloads the RRUFF Raman spectral database and builds a processed dataset.

RRUFF (rruff.info) provides open-access Raman spectra for ~4000+ minerals.
We download the 'excellent' and 'fair' quality zips, parse every spectrum
file, preprocess each one, then save numpy arrays ready for model training.
"""
import os
import re
import sys
import zipfile
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import preprocess_spectrum
from config import STANDARD_GRID, MIN_SAMPLES_PER_CLASS

RRUFF_BASE = "https://rruff.info/zipped_data_files/raman/"
RRUFF_ZIPS = [
    "excellent_unoriented.zip",
    "excellent_oriented.zip",
    "fair_unoriented.zip",
    "fair_oriented.zip",
]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_rruff(data_dir: str = "data") -> Path:
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for zipname in RRUFF_ZIPS:
        zip_path = raw_dir / zipname
        extract_dir = raw_dir / zipname.replace(".zip", "")

        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"  [skip] {zipname} already extracted.")
            continue

        url = RRUFF_BASE + zipname
        print(f"  Downloading {zipname} from {url} ...")
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(zip_path, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc=zipname
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)
                    pbar.update(len(chunk))
            print(f"  Extracting {zipname} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            os.remove(zip_path)
        except Exception as exc:
            print(f"  WARNING: Could not download {zipname}: {exc}")

    return raw_dir


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"##NAMES?\s*=\s*(.+)", re.IGNORECASE)


def _parse_rruff_file(filepath: Path):
    """
    Parse one RRUFF spectrum txt file.

    Supports both legacy format (##END= marker) and new format
    (data begins on the first non-## non-blank line after the header block).

    Returns (mineral_name: str, wavenumbers: ndarray, intensities: ndarray)
    or None on failure.
    """
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    lines = content.splitlines()
    mineral_name = None
    data_lines = []
    in_header = False      # have we seen at least one ## line?
    header_done = False    # have we passed the header block?

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("##"):
            in_header = True
            if not mineral_name:
                m = _NAME_RE.match(stripped)
                if m:
                    mineral_name = m.group(1).split(",")[0].split("/")[0].strip()
            if stripped == "##END=":
                header_done = True
        else:
            # Once we've seen the header block, all non-blank non-## lines are data
            if in_header and stripped:
                data_lines.append(stripped)

    # Fallback: extract name from filename
    # Format: MineralName__RRUFFID__Raman__WL____Raman_Data_Processed__hash.txt
    if not mineral_name:
        parts = filepath.stem.split("__")
        if parts:
            mineral_name = parts[0].replace("_", " ").strip()

    if not mineral_name or not data_lines:
        return None

    wn_list, inten_list = [], []
    for line in data_lines:
        try:
            # Handle both comma-separated and whitespace-separated
            parts = line.replace(";", ",").split(",")
            if len(parts) >= 2:
                w = float(parts[0].strip())
                i = float(parts[1].strip())
                wn_list.append(w)
                inten_list.append(i)
        except ValueError:
            continue

    if len(wn_list) < 20:
        return None

    return mineral_name, np.array(wn_list), np.array(inten_list)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(data_dir: str = "data", min_samples: int = MIN_SAMPLES_PER_CLASS):
    raw_dir = Path(data_dir) / "raw"
    processed_dir = Path(data_dir) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Collect all Raman spectrum files — both Processed and RAW (skip IR files)
    # Including RAW files ~doubles training data and exposes the model to
    # fluorescence-affected spectra that ALS baseline correction must handle.
    all_files = []
    for child in raw_dir.iterdir():
        if child.is_dir():
            for fpath in child.rglob("*.txt"):
                name_lower = fpath.name.lower()
                if "raman" in name_lower and "ir" not in name_lower:
                    all_files.append(fpath)

    # Fallback: accept any .txt if filter yielded nothing
    if not all_files:
        for child in raw_dir.iterdir():
            if child.is_dir():
                all_files.extend(child.rglob("*.txt"))

    print(f"Found {len(all_files)} spectrum files — preprocessing...")

    raw_records = []  # (mineral_name, spectrum_array)
    for fpath in tqdm(all_files, desc="Preprocessing"):
        result = _parse_rruff_file(fpath)
        if result is None:
            continue
        mineral_name, wn, inten = result
        processed = preprocess_spectrum(wn, inten)
        if processed is not None:
            raw_records.append((mineral_name, processed))

    print(f"Successfully preprocessed {len(raw_records)} spectra.")

    if not raw_records:
        raise RuntimeError(
            "No spectra could be processed. Check data/raw/ directory."
        )

    # Count samples per mineral
    from collections import Counter
    counts = Counter(name for name, _ in raw_records)
    valid_minerals = sorted(
        name for name, cnt in counts.items() if cnt >= min_samples
    )
    print(
        f"Minerals with >= {min_samples} spectra: {len(valid_minerals)} "
        f"(dropped {len(counts) - len(valid_minerals)})"
    )

    label_map = {name: i for i, name in enumerate(valid_minerals)}

    X, y, mineral_names = [], [], []
    for name, spectrum in raw_records:
        if name in label_map:
            X.append(spectrum)
            y.append(label_map[name])
            mineral_names.append(name)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    np.save(processed_dir / "X.npy", X)
    np.save(processed_dir / "y.npy", y)

    # Label map CSV
    labels_df = pd.DataFrame({
        "label": range(len(valid_minerals)),
        "mineral": valid_minerals,
    })
    labels_df.to_csv(processed_dir / "label_map.csv", index=False)

    # Reference spectra (mean spectrum per class) for NNLS unmixing
    ref_spectra = np.zeros((len(valid_minerals), X.shape[1]), dtype=np.float32)
    for label, name in enumerate(valid_minerals):
        class_mask = y == label
        if class_mask.sum() > 0:
            mean_spec = X[class_mask].mean(axis=0)
            peak = mean_spec.max()
            ref_spectra[label] = mean_spec / peak if peak > 0 else mean_spec
    np.save(processed_dir / "reference_spectra.npy", ref_spectra)

    print(
        f"\nDataset ready: {len(X)} spectra, {len(valid_minerals)} mineral classes."
    )
    return X, y, label_map
