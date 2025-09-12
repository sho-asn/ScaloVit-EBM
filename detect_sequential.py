import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pathlib import Path
import argparse
import os
from glob import glob
import math
from tqdm import tqdm
import numpy as np

# --- Model & Data Handling Classes (Copied from train_sequential.py) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SequenceTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.d_model = embed_dim

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.input_proj(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)
        output = self.transformer_encoder(src)
        last_token_output = output[:, -1, :]
        prediction = self.output_proj(last_token_output)
        return prediction

class SlidingWindowDataset(Dataset):
    def __init__(self, features: torch.Tensor, sequence_length: int):
        self.features = features
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        return self.features[idx:idx+self.sequence_length], self.features[idx+self.sequence_length]

# --- Other Imports ---
from ebm_model_vit import EBViTModelWrapper as EBM
from metrics import compute_all_metrics
from utils.utils_visualization import plot_energy_with_anomalies

# --- Argument Parsing ---
def get_args():
    parser = argparse.ArgumentParser(description="Evaluate the full two-stage sequential model.")
    parser.add_argument("--stage_a_ckpt_path", type=str, required=True, help="Path to the trained Stage A (EBM) model checkpoint.")
    parser.add_argument("--stage_b_ckpt_path", type=str, required=True, help="Path to the trained Stage B (Sequence) model checkpoint.")
    parser.add_argument("--data_dir", type=str, default="preprocessed_dataset", help="Directory containing the preprocessed chunk files (*.pt).")
    parser.add_argument("--features_dir", type=str, default="./results/features", help="Directory containing the extracted feature files (*.pt).")
    parser.add_argument("--output_dir", type=str, default="./results/final_detection", help="Directory to save plots and results.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing data.")
    parser.add_argument("--threshold_percentile", type=float, default=99.0, help="Percentile of validation scores to use as anomaly threshold.")
    return parser.parse_args()

# --- Main Detection Logic ---

def get_anomaly_scores(model_b, features, sequence_length, batch_size, device, desc):
    """Helper function to compute anomaly scores (MSE) for a feature set."""
    dataset = SlidingWindowDataset(features, sequence_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_scores = []
    criterion = nn.MSELoss(reduction='none')
    with torch.no_grad():
        for x_seq, y_true in tqdm(loader, desc=desc):
            x_seq, y_true = x_seq.to(device), y_true.to(device)
            y_pred = model_b(x_seq)
            score = criterion(y_pred, y_true).mean(dim=1)
            all_scores.extend(score.cpu().numpy())
    return np.array(all_scores)

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    # --- Load Stage B (Sequence) Model and its args ---
    print(f"Loading Stage B checkpoint from {args.stage_b_ckpt_path}...")
    ckpt_b = torch.load(args.stage_b_ckpt_path, map_location=device, weights_only=False)
    model_args_b = ckpt_b['args']
    
    # We need the embedding dim used to train Stage B. We can get it from the checkpoint.
    # A bit of a hack: load the state dict and inspect the first parameter's size.
    embed_dim = ckpt_b['model_state_dict']['input_proj.weight'].shape[1]
    hidden_dim = embed_dim * model_args_b.hidden_dim_multiplier

    model_b = SequenceTransformer(
        embed_dim=embed_dim, num_layers=model_args_b.num_layers, num_heads=model_args_b.num_heads,
        hidden_dim=hidden_dim, dropout=model_args_b.dropout
    ).to(device)
    model_b.load_state_dict(ckpt_b['model_state_dict'])
    model_b.eval()
    print("Stage B model loaded successfully.")

    # --- Establish Anomaly Threshold from Validation Data ---
    print("Processing validation set to establish anomaly threshold...")
    val_features_path = glob(os.path.join(args.features_dir, "*val*.pt"))[0]
    val_features = torch.load(val_features_path)['global_tokens']
    val_scores = get_anomaly_scores(model_b, val_features, model_args_b.sequence_length, args.batch_size, device, "Stage B: Scoring Validation Set")
    anomaly_threshold = np.percentile(val_scores, args.threshold_percentile)
    print(f"Anomaly threshold set to {anomaly_threshold:.6f} ({args.threshold_percentile}th percentile of validation scores)")

    # --- Evaluate on Test Sets ---
    test_feature_files = sorted([f for f in glob(os.path.join(args.features_dir, "*test*.pt"))])
    test_chunk_files = {Path(f).name.replace("features", "chunks"): f for f in sorted([f for f in glob(os.path.join(args.data_dir, "*test*.pt"))])}

    for feature_path in test_feature_files:
        set_name = Path(feature_path).stem
        print(f"\n--- Evaluating {set_name} ---")
        
        # Load features and original chunk data for metadata
        features_data = torch.load(feature_path)
        chunk_data_path = test_chunk_files.get(Path(feature_path).name.replace("features", "chunks"))
        if not chunk_data_path:
            print(f"Warning: Could not find matching chunk file for {feature_path}. Skipping.")
            continue
        original_chunk_data = torch.load(chunk_data_path)

        test_features = features_data['global_tokens']
        ground_truth, signal_len, stride = original_chunk_data['labels'], original_chunk_data['signal_len'], original_chunk_data['stride']
        patch_size = 16 # Assuming patch size, should ideally be in metadata

        # Get anomaly scores for the test set
        test_scores = get_anomaly_scores(model_b, test_features, model_args_b.sequence_length, args.batch_size, device, "Stage B: Scoring Test Set")

        # Create binary predictions
        predicted_anomalies = (test_scores > anomaly_threshold).astype(int)

        # Align scores with ground truth for evaluation
        start_idx = model_args_b.sequence_length
        aligned_gt = ground_truth[start_idx : start_idx + len(test_scores)]

        # Call metrics function with all required arguments
        metrics = compute_all_metrics(aligned_gt.numpy(), predicted_anomalies, test_scores)
        print("Computed Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

if __name__ == "__main__":
    main()