import torch
from pathlib import Path
import argparse
import os
from glob import glob
from tqdm import tqdm

from ebm_model_vit import EBViTModelWrapper as EBM
from torch.utils.data import TensorDataset, DataLoader

def get_args():
    parser = argparse.ArgumentParser(description="Extract features for Stage B model from a trained EBM.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the trained EBM model checkpoint file.")
    parser.add_argument("--data_dir", type=str, default="preprocessed_dataset", help="Directory containing the preprocessed chunk files (*.pt).")
    parser.add_argument("--output_dir", type=str, default="features", help="Directory to save the extracted feature files.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing data.")
    parser.add_argument("--sets", nargs='+', default=["train", "val", "test"], choices=["train", "val", "test"], help="Which data sets to process.")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"Using device: {device}")
    print(f"Saving features to: {output_dir}")

    # --- 1. Load Model from Checkpoint ---
    print(f"Loading checkpoint from {args.ckpt_path}...")
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {args.ckpt_path}")
        
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model_args = ckpt['args']
    
    # Robustly find a sample data file to infer model dimensions
    try:
        potential_files = glob(os.path.join(args.data_dir, "train_*.pt")) + \
                          glob(os.path.join(args.data_dir, "val_*.pt")) + \
                          glob(os.path.join(args.data_dir, "test_*.pt"))
        if not potential_files:
            raise IndexError
        sample_data_path = potential_files[0]
        sample_chunks = torch.load(sample_data_path)
        if isinstance(sample_chunks, dict):
            sample_chunks = sample_chunks['chunks']
        _, C, H, W = sample_chunks.shape
    except IndexError:
        print(f"Error: No train, val, or test data files found in {args.data_dir}. Cannot infer model dimensions.")
        return

    model = EBM(
        dim=(C, H, W),
        num_channels=model_args.num_channels,
        num_res_blocks=model_args.num_res_blocks,
        channel_mult=model_args.channel_mult,
        attention_resolutions=model_args.attention_resolutions,
        num_heads=model_args.num_heads,
        num_head_channels=model_args.num_head_channels,
        dropout=model_args.dropout,
        patch_size=model_args.patch_size,
        embed_dim=model_args.embed_dim,
        transformer_nheads=model_args.transformer_nheads,
        transformer_nlayers=model_args.transformer_nlayers,
        global_energy_weight=getattr(model_args, 'global_energy_weight', 1.0),
        output_scale=model_args.output_scale,
        energy_clamp=model_args.energy_clamp,
    ).to(device)
    
    model.load_state_dict(ckpt['ema_model'])
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Find and Process Data Files ---
    all_files = sorted(glob(os.path.join(args.data_dir, "*.pt")))

    for set_name in args.sets:
        print(f"\n--- Processing {set_name} files ---")
        files_to_process = [f for f in all_files if set_name in Path(f).name]

        if not files_to_process:
            print(f"No files found for set '{set_name}'. Skipping.")
            continue

        for file_path in files_to_process:
            print(f"  Processing file: {Path(file_path).name}")
            original_data = torch.load(file_path)
            
            all_global_tokens = []
            all_patch_embeddings = []

            chunks = original_data['chunks'] if isinstance(original_data, dict) else original_data

            dataset = TensorDataset(chunks)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            with torch.no_grad():
                for batch in tqdm(loader, desc="Extracting"):
                    # Ensure batch is float32, matching the model's weights
                    input_tensor = batch[0].to(device).float()
                    _, global_token, encoded = model.potential(input_tensor, t=None, return_features=True)
                    all_global_tokens.append(global_token.cpu())
                    all_patch_embeddings.append(encoded.cpu())
            
            # Create the final tensors for the entire file
            global_tokens_tensor = torch.cat(all_global_tokens, dim=0)
            patch_embeddings_tensor = torch.cat(all_patch_embeddings, dim=0)

            # --- 3. Save the New Feature Files ---
            base_name = Path(file_path).stem
            new_name = base_name.replace("chunks", "features")
            output_path = output_dir / f"{new_name}.pt"

            # Always save as a dictionary for clarity
            new_data = {
                'global_tokens': global_tokens_tensor,
                'patch_embeddings': patch_embeddings_tensor,
            }

            if isinstance(original_data, dict):
                # For test files, add original metadata
                new_data['labels'] = original_data['labels']
                new_data['signal_len'] = original_data['signal_len']
                new_data['stride'] = original_data['stride']

            torch.save(new_data, output_path)
            print(f"  Saved features to {output_path}")

    print("\nExtraction complete.")

if __name__ == "__main__":
    main()
