import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse
import os
from glob import glob
import math
from tqdm import tqdm
import numpy as np
import copy

from ebm_utils import infiniteloop, ema

# --- 1. New Dataset for Patch-Aware Sequences ---

class SlidingWindowPatchDataset(Dataset):
    """
    Creates a sliding window dataset for the patch-aware sequential model.
    Each item consists of a sequence of chunks and the target chunk that follows.
    A chunk is represented by its global token and all its patch embeddings.
    """
    def __init__(self, feature_files: list, sequence_length: int):
        self.sequence_length = sequence_length
        
        all_global_tokens = []
        all_patch_embeddings = []
        
        print("Loading feature files for dataset...")
        for file_path in tqdm(feature_files):
            data = torch.load(file_path, map_location='cpu')
            all_global_tokens.append(data['global_tokens'])
            all_patch_embeddings.append(data['patch_embeddings'])
            
        self.global_tokens = torch.cat(all_global_tokens, dim=0)
        self.patch_embeddings = torch.cat(all_patch_embeddings, dim=0)
        
        # Combine global token and patch embeddings into a single representation
        # Shape: (num_chunks, 1 + num_patches, embed_dim)
        self.features = torch.cat([
            self.global_tokens.unsqueeze(1),
            self.patch_embeddings
        ], dim=1)
        
        self.num_patches = self.patch_embeddings.shape[1]
        print(f"Dataset loaded. Total chunks: {len(self.features)}, Num patches per chunk: {self.num_patches}")

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        # Input sequence: from idx to idx + sequence_length
        x = self.features[idx:idx+self.sequence_length]
        # Target: the chunk immediately following the sequence
        y = self.features[idx+self.sequence_length]
        return x, y

# --- 2. New Model: Patch-Aware Sequential Transformer ---

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
        # x shape: (S, B, E)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PatchSequenceTransformer(nn.Module):
    """
    A Transformer model that processes a sequence of chunks, where each chunk
    is represented by its global token and all its patch embeddings.
    
    It learns to predict the representation of the next chunk in the sequence.
    """
    def __init__(self, embed_dim: int, num_patches: int, num_layers: int, num_heads: int, hidden_dim_multiplier: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.d_model = embed_dim # Keep model dimension same as embedding dim for simplicity

        # --- Architecture ---
        # 1. Input projection (optional, but good practice)
        self.input_proj = nn.Linear(embed_dim, self.d_model)
        
        # 2. Positional Encoder for the flattened sequence of patches
        # Max len is sequence_length * (1 global token + N patches)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, max_len=50 * (1 + num_patches))

        # 3. Main Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim_multiplier * self.d_model, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. Output Head
        # Takes the final token embedding and predicts the entire next chunk representation
        self.output_proj = nn.Linear(self.d_model, (1 + num_patches) * embed_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src shape: (B, S, 1 + N, E) 
        # B=batch, S=sequence_length, N=num_patches, E=embed_dim
        
        batch_size, seq_len, num_tokens_per_chunk, _ = src.shape
        
        # Flatten the sequence and patch dimensions
        # -> (B, S * (1+N), E)
        src_flat = src.view(batch_size, seq_len * num_tokens_per_chunk, self.embed_dim)
        
        # Project and add positional encoding
        src_proj = self.input_proj(src_flat) * math.sqrt(self.d_model)
        
        # Transformer expects (S, B, E) if batch_first=False (default for PositionalEncoding)
        # Or (B, S, E) if batch_first=True (for TransformerEncoderLayer)
        # Our PositionalEncoding is not batch_first, so we transpose
        src_pe = self.pos_encoder(src_proj.transpose(0, 1)).transpose(0, 1)
        
        # Pass through transformer
        # -> (B, S * (1+N), D) where D=d_model
        output = self.transformer_encoder(src_pe)
        
        # Take the embedding of the very last token of the sequence as the context
        # -> (B, D)
        last_token_context = output[:, -1, :]
        
        # Project this context to predict the (1+N)*E features of the next chunk
        # -> (B, (1+N)*E)
        prediction = self.output_proj(last_token_context)
        
        # Reshape to the structure of a single chunk
        # -> (B, 1+N, E)
        prediction = prediction.view(batch_size, 1 + self.num_patches, self.embed_dim)
        
        return prediction

# --- 3. Training Script ---

def get_args():
    parser = argparse.ArgumentParser(description="Train the Patch-Aware Sequential Model (Stage B).")
    # Data and Paths
    parser.add_argument("--features_dir", type=str, default="./results/features", help="Directory with extracted features from Stage A.")
    parser.add_argument("--output_dir", type=str, default="./results/sequential_patch_training", help="Directory for results and checkpoints.")
    parser.add_argument("--model_name", type=str, default="sequential_patch_model", help="Name for the model and output files.")
    
    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs.")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--sequence_length", type=int, default=10, help="Number of past chunks to use for prediction.")

    # Model Architecture
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the Transformer.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--hidden_dim_multiplier", type=int, default=4, help="Multiplier for hidden dim in feedforward layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")

    # Checkpoint & Logging
    parser.add_argument("--log_step", type=int, default=50, help="Logging frequency.")
    parser.add_argument("--save_epoch", type=int, default=10, help="Checkpoint save frequency.")

    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    # --- Data ---
    train_feature_files = sorted(glob(os.path.join(args.features_dir, "*train*.pt")))
    if not train_feature_files:
        raise FileNotFoundError(f"No training feature files found in {args.features_dir}")
        
    train_dataset = SlidingWindowPatchDataset(train_feature_files, args.sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    # Infer num_patches and embed_dim from the dataset itself for robustness
    num_patches = train_dataset.num_patches
    embed_dim = train_dataset.features.shape[-1]
    print(f"Inferred from data -> Num Patches: {num_patches}, Embedding Dim: {embed_dim}")


    # --- Model ---
    model = PatchSequenceTransformer(
        embed_dim=embed_dim,
        num_patches=num_patches,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim_multiplier=args.hidden_dim_multiplier,
        dropout=args.dropout
    ).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # --- Optimizer, Scheduler, and Loss ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (x_seq, y_true) in enumerate(progress_bar):
            x_seq, y_true = x_seq.to(device), y_true.to(device)
            
            optimizer.zero_grad()
            
            y_pred = model(x_seq)
            
            loss = criterion(y_pred, y_true)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if i % args.log_step == 0:
                progress_bar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

        scheduler.step()

        if (epoch + 1) % args.save_epoch == 0:
            ckpt_path = output_dir / f"{args.model_name}_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                'num_patches': num_patches # Save this for inference
            }, ckpt_path)
            print(f"--- Checkpoint saved to {ckpt_path} ---")

    print("Training finished.")
    final_path = output_dir / f"{args.model_name}_final.pt"
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args,
        'num_patches': num_patches
    }, final_path)
    print(f"--- Final model saved to {final_path} ---")


if __name__ == "__main__":
    main()
