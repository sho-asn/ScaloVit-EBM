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