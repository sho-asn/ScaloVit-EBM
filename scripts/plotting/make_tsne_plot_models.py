import argparse
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
LSTM_SRC = REPO_ROOT / "LSTM-AutoEncoder-Unsupervised-Anomaly-Detection-master" / "src"
PATCHTRAD_ROOT = REPO_ROOT / "PatchTrAD-main"
UTILS_ROOT = REPO_ROOT / "utils"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LSTM_SRC) not in sys.path:
    sys.path.insert(0, str(LSTM_SRC))
if str(PATCHTRAD_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCHTRAD_ROOT))

# Colab environments sometimes preload a third-party `utils` module (a plain
# module rather than a package), which blocks importing the LSTM repo's
# `utils` namespace. Clear that eager import so the namespace package from the
# repository can be discovered on `sys.path`.
if "utils" in sys.modules and not hasattr(sys.modules["utils"], "__path__"):
    del sys.modules["utils"]

if "dataset" in sys.modules and not hasattr(sys.modules["dataset"], "__path__"):
    del sys.modules["dataset"]

if "utils" not in sys.modules or not hasattr(sys.modules["utils"], "__path__"):
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = [str(UTILS_ROOT)]
    sys.modules["utils"] = utils_pkg


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LSTM_MODEL_MOD = _load_module("lstm_model", LSTM_SRC / "model" / "LSTM_auto_encoder.py")
LSTM_DATA_MOD = _load_module("lstm_mfp_dataset", LSTM_SRC / "utils" / "mfp_dataset.py")
PREPROCESS_MOD = _load_module("preprocess_data_mod", REPO_ROOT / "preprocess_data.py")
UTILS_DATA_MOD = _load_module("utils.utils_data", UTILS_ROOT / "utils_data.py")
UTILS_VIS_MOD = _load_module("utils.utils_visualization", UTILS_ROOT / "utils_visualization.py")

LSTMAutoEncoder = LSTM_MODEL_MOD.LSTMAutoEncoder  # type: ignore[attr-defined]
WindowDataset = LSTM_DATA_MOD.WindowDataset  # type: ignore[attr-defined]
build_window_array = LSTM_DATA_MOD.build_window_array  # type: ignore[attr-defined]
compute_feature_stats = LSTM_DATA_MOD.compute_feature_stats  # type: ignore[attr-defined]
load_case_signal = LSTM_DATA_MOD.load_case_signal  # type: ignore[attr-defined]
load_training_signals = LSTM_DATA_MOD.load_training_signals  # type: ignore[attr-defined]
normalize_windows = LSTM_DATA_MOD.normalize_windows  # type: ignore[attr-defined]
sliding_windows = LSTM_DATA_MOD.sliding_windows  # type: ignore[attr-defined]
get_ground_truth_for_signal = PREPROCESS_MOD.get_ground_truth_for_signal  # type: ignore[attr-defined]

# Re-export utilities needed by preprocess or downstream modules when they
# expect `utils` to behave like a package within this runtime session.
sys.modules.setdefault("utils.utils_data", UTILS_DATA_MOD)
sys.modules.setdefault("utils.utils_visualization", UTILS_VIS_MOD)

PATCHTRAD_MOD = _load_module("patchtrad_module", PATCHTRAD_ROOT / "patchtrad.py")
DATASET_MFP_MOD = _load_module("patchtrad_dataset_mfp", PATCHTRAD_ROOT / "dataset" / "mfp.py")

PatchTrad = PATCHTRAD_MOD.PatchTrad  # type: ignore[attr-defined]
load_mfp_datasets = DATASET_MFP_MOD.load_mfp_datasets  # type: ignore[attr-defined]
FAULTY_INTERVALS = DATASET_MFP_MOD.FAULTY_INTERVALS  # type: ignore[attr-defined]


@dataclass
class PatchTradConfig:
    bs: int
    lr: float
    ws: int
    in_dim: int
    epochs: int
    stride: int
    patch_len: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int


FAULT_CASE_IDS = {f"FaultyCase{i}": i for i in range(1, 7)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate t-SNE plots for LSTM autoencoder and PatchTrAD embeddings on the MFP dataset."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("lstm", "patchtrad"),
        default=("lstm", "patchtrad"),
        help="Which models to process.",
    )
    parser.add_argument("--data_dir", type=str, default="Datasets/CVACaseStudy/MFP", help="Path to the MFP dataset root.")
    parser.add_argument("--output_dir", type=str, default="results/plots", help="Directory where plots will be saved.")
    parser.add_argument("--limit_samples_per_class", type=int, default=2000, help="Maximum samples per class for t-SNE (0 disables).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for sampling and t-SNE.")

    # LSTM-specific arguments
    parser.add_argument("--lstm_ckpt", type=str, help="Path to a checkpoint with LSTM state_dict (required when using LSTM).")
    parser.add_argument("--lstm_seq_len", type=int, default=128, help="Window length used by the LSTM model.")
    parser.add_argument("--lstm_train_stride", type=int, default=32, help="Stride for extracting normal windows.")
    parser.add_argument("--lstm_eval_stride", type=int, default=128, help="Stride for extracting evaluation windows.")
    parser.add_argument("--lstm_batch_size", type=int, default=256, help="Batch size when running the encoder.")
    parser.add_argument("--lstm_num_layers", type=int, default=1, help="Fallback LSTM layer count if not stored in checkpoint.")
    parser.add_argument("--lstm_hidden_size", type=int, default=128, help="Fallback LSTM hidden size if not stored in checkpoint.")
    parser.add_argument("--lstm_dropout", type=float, default=0.0, help="Fallback LSTM dropout if not stored in checkpoint.")

    # PatchTrAD-specific arguments
    parser.add_argument("--patchtrad_ckpt", type=str, help="Path to PatchTrAD checkpoint (required when using PatchTrAD).")
    parser.add_argument("--patchtrad_window_size", type=int, default=128, help="Sliding-window size used by PatchTrAD (ws).")
    parser.add_argument("--patchtrad_stride", type=int, default=16, help="Stride used during PatchTrAD training.")
    parser.add_argument("--patchtrad_batch_size", type=int, default=256, help="Batch size for PatchTrAD embedding extraction.")
    parser.add_argument("--patchtrad_d_model", type=int, default=64, help="Fallback transformer width if not stored in checkpoint.")
    parser.add_argument("--patchtrad_n_heads", type=int, default=4, help="Fallback number of attention heads if not stored in checkpoint.")
    parser.add_argument("--patchtrad_n_layers", type=int, default=3, help="Fallback transformer depth if not stored in checkpoint.")
    parser.add_argument("--patchtrad_d_ff", type=int, default=128, help="Fallback feed-forward width if not stored in checkpoint.")
    parser.add_argument("--patchtrad_patch_len", type=int, default=16, help="Patch length used inside PatchTrAD.")
    parser.add_argument("--patchtrad_lr", type=float, default=1e-4, help="Dummy learning rate to satisfy config field expectations.")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_by_class(X: np.ndarray, y: np.ndarray, limit: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if limit <= 0:
        return X, y
    samples_X: List[np.ndarray] = []
    samples_y: List[np.ndarray] = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        if idx.size == 0:
            continue
        if idx.size > limit:
            chosen = rng.choice(idx, size=limit, replace=False)
        else:
            chosen = idx
        samples_X.append(X[chosen])
        samples_y.append(y[chosen])
    return np.concatenate(samples_X, axis=0), np.concatenate(samples_y, axis=0)


def run_tsne_and_plot(
    X: np.ndarray,
    y: np.ndarray,
    label_map: Dict[int, str],
    output_path: Path,
    seed: int,
    title: str,
) -> None:
    if X.shape[0] < 2:
        raise ValueError("Need at least two samples to run t-SNE.")
    perplexity = max(5, min(40, X.shape[0] // 10))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, max_iter=1000, verbose=1, init="pca")
    coords = tsne.fit_transform(X)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 22,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "legend.fontsize": 22,
        "legend.title_fontsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.title_fontsize": 22,
        "legend.fontsize": 22,
        # improve resolution of tick label rendering on large fonts
        "xtick.major.size": 6,
        "ytick.major.size": 6,
    })
    fig, ax = plt.subplots(figsize=(14, 12))
    cmap = plt.get_cmap("tab10")
    for label, name in sorted(label_map.items()):
        mask = y == label
        if not np.any(mask):
            continue
        color = cmap(label % 10)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=26, alpha=0.65, label=name, color=color)
    ax.set_title(title)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    # Make tick numbers larger and bolder
    ax.tick_params(axis="both", which="major", labelsize=20, width=1.2)
    # Keep legend in the upper-right even when fonts increase
    ax.legend(loc="upper right", markerscale=2, frameon=True)
    ax.grid(True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved t-SNE plot to {output_path}")


def load_lstm_config_from_ckpt(ckpt: Dict[str, object], fallback: argparse.Namespace) -> Dict[str, object]:
    config = {}
    raw_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    if isinstance(raw_cfg, dict):
        config.update(raw_cfg)
    config.setdefault("num_layers", fallback.lstm_num_layers)
    config.setdefault("hidden_size", fallback.lstm_hidden_size)
    config.setdefault("dropout", fallback.lstm_dropout)
    return config


def extract_lstm_embeddings(args: argparse.Namespace, device: torch.device) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    if not args.lstm_ckpt:
        raise ValueError("--lstm_ckpt is required when requesting LSTM t-SNE plot.")

    data_dir = Path(args.data_dir)
    train_signals = load_training_signals(data_dir)
    mean, std = compute_feature_stats(train_signals)
    train_windows = build_window_array(train_signals, window=args.lstm_seq_len, stride=args.lstm_train_stride)
    if train_windows.size == 0:
        raise ValueError("No training windows generated for LSTM; check seq_len and stride.")
    train_windows = normalize_windows(train_windows, mean, std)

    nb_feature = train_windows.shape[-1]
    ckpt = torch.load(args.lstm_ckpt, map_location=device)
    lstm_cfg = load_lstm_config_from_ckpt(ckpt if isinstance(ckpt, dict) else {}, args)

    model = LSTMAutoEncoder(
        num_layers=int(lstm_cfg["num_layers"]),
        hidden_size=int(lstm_cfg["hidden_size"]),
        nb_feature=nb_feature,
        dropout=float(lstm_cfg.get("dropout", 0.0)),
        device=device,
    ).to(device)

    state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    if state_dict is None:
        if isinstance(ckpt, dict):
            # Supports checkpoints saved via torch.save({'model': state_dict})
            for key in ("model", "ema_model", "ema_state", "lstm_state"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state_dict = ckpt[key]
                    break
        if state_dict is None and isinstance(ckpt, dict):
            state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        if state_dict is None and isinstance(ckpt, dict):
            raise ValueError("Could not locate LSTM state_dict inside checkpoint.")
    if state_dict is None:
        state_dict = ckpt  # checkpoint is already a state_dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[LSTM] Missing keys when loading state_dict: {missing}")
    if unexpected:
        print(f"[LSTM] Unexpected keys when loading state_dict: {unexpected}")
    model.eval()

    label_map = {0: "Normal"}
    rng = np.random.default_rng(args.seed)

    def encode_windows(windows: np.ndarray) -> np.ndarray:
        dataset = WindowDataset(windows)
        loader = DataLoader(dataset, batch_size=args.lstm_batch_size, shuffle=False)
        embeddings: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                hidden, _ = model.encoder(batch)
                # Use the top-layer hidden state as the embedding
                last_hidden = hidden[-1]
                embeddings.append(last_hidden.cpu().numpy())
        if not embeddings:
            return np.empty((0, int(lstm_cfg["hidden_size"])), dtype=np.float32)
        return np.concatenate(embeddings, axis=0)

    normal_embeddings = encode_windows(train_windows)
    normal_labels = np.zeros(normal_embeddings.shape[0], dtype=np.int32)

    all_embeddings = [normal_embeddings]
    all_labels = [normal_labels]

    fault_cases: Sequence[Tuple[str, str, Sequence[Tuple[int, int]]]] = FAULTY_INTERVALS
    for case_file, set_name, intervals in fault_cases:
        try:
            signal = load_case_signal(data_dir, case_file, set_name)
        except (FileNotFoundError, KeyError) as exc:
            print(f"[LSTM] Skipping {case_file}/{set_name}: {exc}")
            continue
        windows, _ = sliding_windows(signal, args.lstm_seq_len, args.lstm_eval_stride)
        if windows.size == 0:
            print(f"[LSTM] No windows generated for {case_file}/{set_name}, skipping.")
            continue
        normalized = normalize_windows(windows, mean, std)
        gt = get_ground_truth_for_signal(intervals, signal.shape[0], args.lstm_seq_len).numpy()
        effective = min(normalized.shape[0], gt.shape[0])
        if effective == 0:
            continue
        mask = gt[:effective] == 1
        if not np.any(mask):
            print(f"[LSTM] No anomalous windows for {case_file}/{set_name}.")
            continue
        embeddings = encode_windows(normalized[:effective][mask])
        if embeddings.size == 0:
            continue
        case_id = FAULT_CASE_IDS.get(case_file, None)
        if case_id is None:
            continue
        label_map.setdefault(case_id, f"Faulty {case_id}")
        labels = np.full(embeddings.shape[0], case_id, dtype=np.int32)
        all_embeddings.append(embeddings)
        all_labels.append(labels)

    X = np.concatenate(all_embeddings, axis=0)
    y = np.concatenate(all_labels, axis=0)
    X, y = limit_by_class(X, y, args.limit_samples_per_class, rng)
    return X, y, label_map


def load_patchtrad_config_from_ckpt(ckpt: Dict[str, object], fallback: argparse.Namespace) -> Dict[str, object]:
    config = {}
    raw_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    if isinstance(raw_cfg, dict):
        config.update(raw_cfg)
    for key, value in (
        ("d_model", fallback.patchtrad_d_model),
        ("n_heads", fallback.patchtrad_n_heads),
        ("n_layers", fallback.patchtrad_n_layers),
        ("d_ff", fallback.patchtrad_d_ff),
        ("patch_len", fallback.patchtrad_patch_len),
        ("stride", fallback.patchtrad_stride),
        ("ws", fallback.patchtrad_window_size),
    ):
        config.setdefault(key, value)
    return config


def extract_patchtrad_embeddings(args: argparse.Namespace, device: torch.device) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    if not args.patchtrad_ckpt:
        raise ValueError("--patchtrad_ckpt is required when requesting PatchTrAD t-SNE plot.")

    train_dataset, test_datasets, _, _, _ = load_mfp_datasets(
        data_dir=args.data_dir,
        window_size=args.patchtrad_window_size,
        step=args.patchtrad_stride,
    )

    sample_window, _ = train_dataset[0]
    in_dim = sample_window.shape[-1]
    ckpt = torch.load(args.patchtrad_ckpt, map_location=device)
    patchtrad_cfg = load_patchtrad_config_from_ckpt(ckpt if isinstance(ckpt, dict) else {}, args)
    config = PatchTradConfig(
        bs=args.patchtrad_batch_size,
        lr=args.patchtrad_lr,
        ws=int(patchtrad_cfg["ws"]),
        in_dim=in_dim,
        epochs=0,
        stride=int(patchtrad_cfg["stride"]),
        patch_len=int(patchtrad_cfg["patch_len"]),
        d_model=int(patchtrad_cfg["d_model"]),
        n_heads=int(patchtrad_cfg["n_heads"]),
        n_layers=int(patchtrad_cfg["n_layers"]),
        d_ff=int(patchtrad_cfg["d_ff"]),
    )

    model = PatchTrad(config).to(device)
    state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    if state_dict is None:
        for key in ("model", "ema_model", "ema_state"):
            if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        if state_dict is None and isinstance(ckpt, dict):
            state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        if state_dict is None:
            state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[PatchTrAD] Missing keys when loading state_dict: {missing}")
    if unexpected:
        print(f"[PatchTrAD] Unexpected keys when loading state_dict: {unexpected}")
    model.eval()

    rng = np.random.default_rng(args.seed)
    label_map = {0: "Normal"}

    def encode_loader(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        embeddings: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        with torch.no_grad():
            for windows, targets in loader:
                windows = windows.to(device)
                patched = model._get_patch(windows)
                encoded = model.encoder(patched)
                last_patch = encoded[:, :, :, -1]
                flat = last_patch.reshape(last_patch.shape[0], -1).cpu().numpy()
                embeddings.append(flat)
                labels.append(targets.numpy())
        if not embeddings:
            feature_dim = model.head_layer.n_vars * config.d_model
            return np.empty((0, feature_dim), dtype=np.float32), np.empty((0,), dtype=np.int32)
        return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)

    normal_loader = DataLoader(train_dataset, batch_size=args.patchtrad_batch_size, shuffle=False)
    normal_embeddings, _ = encode_loader(normal_loader)
    normal_labels = np.zeros(normal_embeddings.shape[0], dtype=np.int32)

    all_embeddings = [normal_embeddings]
    all_labels = [normal_labels]

    for key, dataset in test_datasets.items():
        parts = key.split("_")
        if not parts:
            continue
        case_id = FAULT_CASE_IDS.get(parts[0], None)
        if case_id is None:
            continue
        loader = DataLoader(dataset, batch_size=args.patchtrad_batch_size, shuffle=False)
        embeddings, labels = encode_loader(loader)
        if embeddings.size == 0 or labels.size == 0:
            continue
        mask = labels == 1
        if not np.any(mask):
            continue
        label_map.setdefault(case_id, f"Faulty {case_id}")
        all_embeddings.append(embeddings[mask])
        all_labels.append(np.full(mask.sum(), case_id, dtype=np.int32))

    X = np.concatenate(all_embeddings, axis=0)
    y = np.concatenate(all_labels, axis=0)
    X, y = limit_by_class(X, y, args.limit_samples_per_class, rng)
    return X, y, label_map


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    selected_models = set(args.models)
    output_dir = Path(args.output_dir)

    if "lstm" in selected_models:
        X_lstm, y_lstm, labels_lstm = extract_lstm_embeddings(args, device)
        plot_path = output_dir / "tsne_lstm_autoencoder.png"
        run_tsne_and_plot(X_lstm, y_lstm, labels_lstm, plot_path, args.seed, "LSTM Autoencoder Embeddings")

    if "patchtrad" in selected_models:
        X_pt, y_pt, labels_pt = extract_patchtrad_embeddings(args, device)
        plot_path = output_dir / "tsne_patchtrad.png"
        run_tsne_and_plot(X_pt, y_pt, labels_pt, plot_path, args.seed, "PatchTrAD Embeddings")


if __name__ == "__main__":
    main()
