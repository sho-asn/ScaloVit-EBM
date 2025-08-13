import torch

# init the min and max values for the STFTEmbedder, this function must be called before the training loop starts
def init_stft_embedder(embedder, train_loader):
    """
    Initializes min/max values for normalization across the whole dataset.
    Args:
        embedder (STFTEmbedder): the embedder object.
        train_loader (DataLoader): training data loader.
    """
    data = []
    for data_batch in train_loader:
        data.append(data_batch[0])  # Extract input tensor from batch tuple
    all_data = torch.cat(data, dim=0)  # Concatenate along batch dimension
    print(all_data.shape)
    embedder.cache_min_max_params(torch.cat(data, dim=0))

def init_wav_embedder(embedder, full_train_signal_np):
    """
    Initializes min/max values for normalization across the whole training signal.
    Args:
        embedder (WAVEmbedder_ST): the embedder object.
        full_train_signal_np (np.ndarray): The entire training time-series data as numpy array.
                                             Shape: (1, L, F) for single batch of full signal
    """
    # Use a dummy signal_length for the embedder's internal seq_len during init,
    # as we're interested in the entire signal's properties.
    # The actual seq_len for the embedder will be updated when ts_to_img is called with the full signal.
    # For caching, we just need the statistics from a representative dataset.
    embedder.cache_min_max_params(full_train_signal_np)

def split_image_into_chunks(image: torch.Tensor, chunk_width: int) -> torch.Tensor:
    """
    Splits a 4D image tensor (B, C, H, W) into chunks along the Width (W) dimension.

    Args:
        image (torch.Tensor): The input 4D tensor representing the wavelet image.
                              Shape: (B, C, H, W) where B is batch, C is channels (3 for WAV),
                              H is height (scales), W is width (time steps).
        chunk_width (int): The desired width for each chunk.

    Returns:
        torch.Tensor: A new tensor containing the image chunks.
                      Shape: (B * num_chunks, C, H, chunk_width).
    """
    batch_size, channels, height, width = image.shape
    num_chunks = width // chunk_width

    # Ensure the image width is a multiple of chunk_width by truncating if necessary
    truncated_width = num_chunks * chunk_width
    image = image[:, :, :, :truncated_width]

    # Reshape: (B, C, H, num_chunks * chunk_width) -> (B, C, H, num_chunks, chunk_width)
    # Then permute and reshape to get (B * num_chunks, C, H, chunk_width)
    chunks = image.reshape(batch_size, channels, height, num_chunks, chunk_width)
    chunks = chunks.permute(0, 3, 1, 2, 4).contiguous() # Reorder to (B, num_chunks, C, H, chunk_width)
    chunks = chunks.view(-1, channels, height, chunk_width) # Flatten B and num_chunks

    return chunks

def get_full_signal_from_dataloader(dataloader) -> torch.Tensor:
    """
    Concatenates all chunks from a DataLoader to form the full signal.
    Assumes the dataloader yields (signal_chunk,)
    """
    all_chunks = []
    for batch_data in dataloader:
        all_chunks.append(batch_data[0])
    # Concatenate along the sequence length dimension (dim=1)
    full_signal = torch.cat(all_chunks, dim=1)
    print(full_signal.shape)
    return full_signal