import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import os
import glob
from torch.utils.data import TensorDataset, DataLoader

# Assuming ebm_model_vit is in the same directory or accessible
from ebm_model_vit import EBViTModelWrapper as EBM

def get_args():
    parser = argparse.ArgumentParser(description="Generate a t-SNE plot of patch embeddings from an EBM model.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--train_data_path", type=str, default="preprocessed_dataset/train_chunks_wavelet_mag.pt", help="Path to the training data (for normal patches).")
    parser.add_argument("--test_data_dir", type=str, default="preprocessed_dataset", help="Directory containing the preprocessed test set files (*.pt).")
    parser.add_argument("--output_path", type=str, default="results/plots/tsne_patch_visualization.png", help="Path to save the output t-SNE plot.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction.")
    parser.add_argument("--limit_samples_per_class", type=int, default=2000, help="Limit the number of samples per class to speed up t-SNE. 0 for no limit.")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model ---
    print(f"Loading checkpoint from {args.ckpt_path}...")
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt['args']
    
    # Use training data to determine model input shape
    train_chunks_for_shape = torch.load(args.train_data_path, map_location='cpu')
    _B, C, H, W = train_chunks_for_shape.shape

    model = EBM(
        dim=(C, H, W),
        num_channels=train_args.num_channels, num_res_blocks=train_args.num_res_blocks,
        channel_mult=train_args.channel_mult, attention_resolutions=train_args.attention_resolutions,
        num_heads=train_args.num_heads, num_head_channels=train_args.num_head_channels,
        dropout=train_args.dropout, patch_size=train_args.patch_size,
        embed_dim=train_args.embed_dim, transformer_nheads=train_args.transformer_nheads,
        transformer_nlayers=train_args.transformer_nlayers, output_scale=train_args.output_scale,
        energy_clamp=train_args.energy_clamp,
    ).to(device)
    model.load_state_dict(ckpt['ema_model'])
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Extract Patch Embeddings ---
    all_patch_embeddings_list = []
    all_patch_labels_list = []
    # Match label format used in make_tsne_plot_models.py
    label_map = {0: 'Normal'}
    for i in range(1, 6):
        label_map[i] = f'Faulty {i}'

    # Determine file suffix from train_data_path
    train_path_name = Path(args.train_data_path).name
    base_name = train_path_name.replace('train_chunks', '')
    file_suffix = base_name.replace('.pt', '')
    print(f"Inferred file suffix as: '{file_suffix}'")

    patch_width = train_args.patch_size[1]

    # Category 0: Normal Patches from Training Data
    print("\nProcessing Normal patches from training data...")
    normal_loader = DataLoader(TensorDataset(train_chunks_for_shape), batch_size=args.batch_size)
    normal_patches_list = []
    with torch.no_grad():
        for batch in normal_loader:
            _, patch_embeddings = model.potential(batch[0].to(device), t=torch.ones(batch[0].size(0), device=device), return_tokens=True)
            num_chunks, num_patches, embed_dim = patch_embeddings.shape
            normal_patches_list.append(patch_embeddings.reshape(num_chunks * num_patches, embed_dim).cpu())
    
    normal_patches = torch.cat(normal_patches_list, dim=0).numpy()
    if args.limit_samples_per_class > 0 and len(normal_patches) > args.limit_samples_per_class:
        indices = np.random.choice(len(normal_patches), args.limit_samples_per_class, replace=False)
        normal_patches = normal_patches[indices]

    all_patch_embeddings_list.append(normal_patches)
    all_patch_labels_list.append(np.full(len(normal_patches), 0, dtype=int))
    print(f"  Extracted {len(normal_patches)} normal patches.")

    # Categories 1-5: Anomaly Patches from Test Data
    for case_idx in range(1, 6):
        print(f"\nProcessing anomaly patches for Faulty Case {case_idx}...")
        file_pattern = f"test_FaultyCase{case_idx}_*{file_suffix}.pt"
        case_files = sorted(glob.glob(os.path.join(args.test_data_dir, file_pattern)))

        if not case_files:
            print(f"  No files found for Faulty Case {case_idx}. Skipping.")
            continue

        case_anomaly_embeddings = []
        for f_path in case_files:
            data = torch.load(f_path, map_location='cpu')
            chunks, global_patch_labels, stride = data['chunks'], data['labels'], data['stride']
            
            loader = DataLoader(TensorDataset(chunks), batch_size=args.batch_size)
            chunk_offset = 0
            with torch.no_grad():
                for batch_chunks in loader:
                    _, batch_patch_embeds = model.potential(batch_chunks[0].to(device), t=torch.ones(batch_chunks[0].size(0), device=device), return_tokens=True)
                    
                    for i in range(batch_chunks[0].shape[0]):
                        chunk_idx = chunk_offset + i
                        start_patch_idx = (chunk_idx * stride) // patch_width
                        num_patches_in_chunk = batch_patch_embeds.shape[1]
                        end_patch_idx = start_patch_idx + num_patches_in_chunk
                        
                        labels = global_patch_labels[start_patch_idx:end_patch_idx]
                        embeds = batch_patch_embeds[i].cpu()

                        min_len = min(len(labels), len(embeds))
                        anomaly_mask = (labels[:min_len] == 1)
                        
                        if torch.any(anomaly_mask):
                            case_anomaly_embeddings.append(embeds[:min_len][anomaly_mask].numpy())
                    chunk_offset += batch_chunks[0].shape[0]

        if not case_anomaly_embeddings:
            print(f"  No anomaly patches found for Faulty Case {case_idx}.")
            continue

        anomaly_patches = np.concatenate(case_anomaly_embeddings, axis=0)
        if args.limit_samples_per_class > 0 and len(anomaly_patches) > args.limit_samples_per_class:
            indices = np.random.choice(len(anomaly_patches), args.limit_samples_per_class, replace=False)
            anomaly_patches = anomaly_patches[indices]
        
        all_patch_embeddings_list.append(anomaly_patches)
        all_patch_labels_list.append(np.full(len(anomaly_patches), case_idx, dtype=int))
        print(f"  Extracted {len(anomaly_patches)} anomaly patches for Faulty Case {case_idx}.")

    # --- 3. Run t-SNE and Plot ---
    X = np.concatenate(all_patch_embeddings_list, axis=0)
    y = np.concatenate(all_patch_labels_list, axis=0)

    print("\nShuffling data to remove temporal ordering for visualization...")
    p = np.random.permutation(len(X))
    X = X[p]
    y = y[p]

    print(f"\nRunning t-SNE on {X.shape[0]} samples with {X.shape[1]} dimensions...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, max_iter=300, random_state=42, n_jobs=-1)
    X_2d = tsne.fit_transform(X)
    print("t-SNE finished.")

    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        "font.size": 22,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "legend.fontsize": 22,
        "legend.title_fontsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        # improve resolution of tick label rendering on large fonts
        "xtick.major.size": 6,
        "ytick.major.size": 6,
    })
    fig, ax = plt.subplots(figsize=(14, 12))

    unique_labels = sorted(label_map.keys())
    cmap = plt.get_cmap('tab10')

    for label_val in unique_labels:
        label_name = label_map.get(label_val)
        if label_name is None:
            continue
        indices = (y == label_val)
        if np.any(indices):
            ax.scatter(
                X_2d[indices, 0], X_2d[indices, 1],
                color=cmap(label_val % 10),
                label=label_name,
                alpha=0.65,
                s=26,
            )

    ax.set_title('t-SNE Visualization of Patch Embeddings')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    # Make tick numbers larger and bolder
    ax.tick_params(axis='both', which='major', labelsize=20, width=1.2)
    # Keep legend in the upper-right corner and match label style (no legend title)
    ax.legend(loc='upper right', markerscale=2, frameon=True)
    ax.grid(True)
    
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {args.output_path}")

if __name__ == "__main__":
    main()
