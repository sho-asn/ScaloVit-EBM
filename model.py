import torch
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from img_transformations import WAVEmbedder

def init_wav_embedder(embedder: "WAVEmbedder", full_train_signal_np: np.ndarray) -> None:
    """
    Initializes min/max values for normalization across the whole training signal.
    Args:
        embedder (WAVEmbedder): the embedder object.
        full_train_signal_np (np.ndarray): The entire training time-series data as a numpy array.
                                             Shape: (1, L, F) for a single batch of the full signal.
    """
    embedder.cache_min_max_params(full_train_signal_np)

def split_image_into_chunks(image: torch.Tensor, chunk_width: int) -> torch.Tensor:
    """
    Splits a 4D image tensor (B, C, H, W) into chunks along the Width (W) dimension.

    Args:
        image (torch.Tensor): The input 4D tensor representing the wavelet image.
                              Shape: (B, C, H, W) where B is batch, C is channels,
                              H is height (scales), W is width (time steps).
        chunk_width (int): The desired width for each chunk.

    Returns:
        torch.Tensor: A new tensor containing the image chunks.
                      Shape: (B * num_chunks, C, H, chunk_width).
    """
    batch_size, channels, height, width = image.shape
    num_chunks = width // chunk_width

    if num_chunks == 0:
        return torch.empty(0, channels, height, chunk_width)

    # Ensure the image width is a multiple of chunk_width by truncating if necessary
    truncated_width = num_chunks * chunk_width
    image = image[:, :, :, :truncated_width]

    # Reshape: (B, C, H, num_chunks * chunk_width) -> (B, C, H, num_chunks, chunk_width)
    # Then permute and reshape to get (B * num_chunks, C, H, chunk_width)
    chunks = image.reshape(batch_size, channels, height, num_chunks, chunk_width)
    chunks = chunks.permute(0, 3, 1, 2, 4).contiguous() # Reorder to (B, num_chunks, C, H, chunk_width)
    chunks = chunks.view(-1, channels, height, chunk_width) # Flatten B and num_chunks

    return chunks
