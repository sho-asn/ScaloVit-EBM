import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse
import os
import math
from tqdm import tqdm

# --- Model Architecture ---

class PositionalEncoding(nn.Module):
    """Injects positional information into the input sequence."""
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
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SequenceTransformer(nn.Module):
    """
    A Transformer model to process sequences of chunk embeddings.
    It's trained on a next-token-prediction style task.
    """
    def __init__(self, embed_dim: int, num_layers: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.input_proj = nn.Linear(embed_dim, embed_dim) # Project input to match model dim
        self.output_proj = nn.Linear(embed_dim, embed_dim) # Project output to predict next embedding
        self.d_model = embed_dim

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        src = self.input_proj(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)
        output = self.transformer_encoder(src)
        # We only care about the output of the last token in the sequence to predict the next one
        last_token_output = output[:, -1, :]
        prediction = self.output_proj(last_token_output)
        return prediction

# --- Data Handling ---

class SequentialFeaturesDataset(Dataset):
    """Creates a dataset of sliding windows from a sequence of features."""
    def __init__(self, features_path: str, sequence_length: int):
        data = torch.load(features_path)
        # Use global tokens as the primary sequence feature
        self.features = data['global_tokens']
        self.sequence_length = sequence_length

        # Create sliding windows
        # X will be a sequence of `sequence_length`
        # y will be the single next item in the sequence
        self.X, self.y = self.create_windows()

    def create_windows(self):
        X, y = [], []
        num_embeddings = len(self.features)
        for i in range(num_embeddings - self.sequence_length):
            X.append(self.features[i:i+self.sequence_length])
            y.append(self.features[i+self.sequence_length])
        return torch.stack(X), torch.stack(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Argument Parsing ---

def get_args():
    parser = argparse.ArgumentParser(description="Train a Stage B sequence model on extracted features.")
    # Paths
    parser.add_argument("--train_features_path", type=str, required=True, help="Path to the training feature file (*.pt).")
    parser.add_argument("--val_features_path", type=str, required=True, help="Path to the validation feature file (*.pt).")
    parser.add_argument("--output_dir", type=str, default="./results/stage_b_training", help="Directory to save model checkpoints.")
    parser.add_argument("--model_name", type=str, default="StageB_Transformer", help="Name for the model and output files.")

    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (number of sequences).")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--sequence_length", type=int, default=32, help="Length of chunk sequences for the model.")

    # Model Architecture
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the Transformer.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in the Transformer.")
    parser.add_argument("--hidden_dim_multiplier", type=int, default=4, help="Multiplier for the transformer's feed-forward layer dimension.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")

    return parser.parse_args()

# --- Main Training & Validation Logic ---

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("--- Preparing Datasets ---")
    train_dataset = SequentialFeaturesDataset(args.train_features_path, args.sequence_length)
    val_dataset = SequentialFeaturesDataset(args.val_features_path, args.sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Infer embedding dimension from the data
    embed_dim = train_dataset.features.shape[1]
    hidden_dim = embed_dim * args.hidden_dim_multiplier
    print(f"Detected embedding dimension: {embed_dim}")

    print("--- Initializing Model ---")
    model = SequenceTransformer(
        embed_dim=embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=hidden_dim,
        dropout=args.dropout
    ).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    print("--- Starting Training ---")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                prediction = model(x_batch)
                loss = criterion(prediction, y_batch)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = output_dir / f"{args.model_name}_best.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'args': args
            }, save_path)
            print(f"New best model saved to {save_path} with validation loss {best_val_loss:.6f}")

    print("--- Training Complete ---")

if __name__ == "__main__":
    main()
