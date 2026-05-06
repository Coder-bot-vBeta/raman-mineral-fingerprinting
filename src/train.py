"""
Training script for RamanNet.

Outputs:
  models/best_model.pth       — best validation checkpoint
  models/training_info.json   — hyperparams + final metrics
  models/history.npy          — loss / accuracy curves
"""
import json
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from model import RamanNet
from dataset import RamanDataset


def _per_class_split(X, y, test_frac=0.15, val_frac=0.15, seed=42):
    """Stratified split that guarantees >= 1 sample per class in val and test."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        # never leave train empty
        n_test = min(n_test, n - 2)
        n_val = min(n_val, n - n_test - 1)
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test:n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def train(
    data_dir: str = "data",
    model_dir: str = "models",
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 25,
):
    processed_dir = Path(data_dir) / "processed"
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    # ---- Load data -------------------------------------------------------
    X = np.load(processed_dir / "X.npy")
    y = np.load(processed_dir / "y.npy")
    n_classes = int(y.max()) + 1
    print(f"Dataset: {X.shape[0]} spectra | {n_classes} mineral classes")

    # Per-class stratified 70 / 15 / 15 split (handles classes with few samples)
    train_idx, val_idx, test_idx = _per_class_split(X, y)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]
    print(
        f"Split: train={len(X_train)} | val={len(X_val)} | test={len(X_test)}"
    )

    # ---- Weighted sampler to handle class imbalance ----------------------
    class_counts = np.bincount(y_train, minlength=n_classes).astype(float)
    class_weights = 1.0 / np.where(class_counts > 0, class_counts, 1.0)
    sample_weights = torch.tensor(
        [class_weights[label] for label in y_train], dtype=torch.float32
    )
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(y_train), replacement=True
    )

    train_ds = RamanDataset(X_train, y_train, augment=True)
    val_ds = RamanDataset(X_val, y_val, augment=False)
    test_ds = RamanDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ---- Model -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = RamanNet(n_classes).to(device)

    # Class-weighted cross-entropy with label smoothing (0.1) to prevent
    # overconfident predictions on visually similar classes (e.g. Calcite vs Rhodochrosite)
    cw = torch.tensor(class_weights / class_weights.sum() * n_classes, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ---- Training loop ---------------------------------------------------
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * len(yb)
            t_correct += (out.argmax(1) == yb).sum().item()
            t_total += len(yb)

        # --- Validate ---
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                v_loss += criterion(out, yb).item() * len(yb)
                v_correct += (out.argmax(1) == yb).sum().item()
                v_total += len(yb)

        train_acc = t_correct / t_total
        val_acc = v_correct / v_total
        history["train_loss"].append(t_loss / t_total)
        history["val_loss"].append(v_loss / v_total)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{epochs}  "
                f"loss={t_loss/t_total:.4f}  train_acc={train_acc:.3f}  "
                f"val_acc={val_acc:.3f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path / "best_model.pth")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
                break

    # ---- Test evaluation -------------------------------------------------
    model.load_state_dict(
        torch.load(model_path / "best_model.pth", map_location=device)
    )
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            test_correct += (out.argmax(1) == yb).sum().item()
            test_total += len(yb)
    test_acc = test_correct / test_total
    print(f"\n{'='*50}")
    print(f"  Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Best Val Acc  : {best_val_acc:.4f}  ({best_val_acc*100:.2f}%)")
    print(f"{'='*50}\n")

    # Save results
    info = {
        "n_classes": n_classes,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "test_accuracy": round(float(test_acc), 4),
        "best_val_accuracy": round(float(best_val_acc), 4),
        "epochs_trained": len(history["train_acc"]),
        "batch_size": batch_size,
        "lr": lr,
    }
    with open(model_path / "training_info.json", "w") as fh:
        json.dump(info, fh, indent=2)

    np.save(model_path / "history.npy", history)
    print("Model saved to models/best_model.pth")
    return model, test_acc, history


if __name__ == "__main__":
    train()
