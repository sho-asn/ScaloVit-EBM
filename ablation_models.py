# File: ablation_models.py
# Contains model variants for ablation studies.

import torch
import torch.nn as nn
import inspect
from ebm_model_vit import PatchEmbed, dummy_time # Reuse helpers from your original file
from torchcfm.models.unet.unet import UNetModelWrapper

# --- Helper function to filter kwargs ---
def filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    allowed_keys = {p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD or p.kind == p.KEYWORD_ONLY}
    return {k: v for k, v in kwargs.items() if k in allowed_keys}

# --- Ablation Model 1: Image-Based Scoring (Global Score per Chunk) ---

class ImageBased_EBViTModelWrapper(UNetModelWrapper):
    """
    This model produces ONE energy score per image chunk.
    """
    def __init__(self, **kwargs):
        unet_kwargs = filter_kwargs(UNetModelWrapper.__init__, kwargs)
        super().__init__(**unet_kwargs)
        
        dim = kwargs['dim']
        patch_size = kwargs['patch_size']
        embed_dim = kwargs['embed_dim']
        transformer_nheads = kwargs['transformer_nheads']
        transformer_nlayers = kwargs['transformer_nlayers']
        dropout = kwargs.get('dropout', 0.1)
        include_pos_embed = kwargs.get('include_pos_embed', True)
        
        self.out_channels = dim[0]
        self.output_scale = kwargs.get('output_scale', 1000.0)
        self.energy_clamp = kwargs.get('energy_clamp')

        self.patch_embed = PatchEmbed(
            in_channels=self.out_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            image_size=dim[1:],
            include_pos_embed=include_pos_embed
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=transformer_nheads,
            dim_feedforward=4 * embed_dim, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_nlayers)
        self.final_linear = nn.Linear(embed_dim, 1)

    def potential(self, x, t, **kwargs):
        t_dummy = dummy_time(x, value=0.5)
        unet_out = super().forward(t_dummy, x)
        tokens = self.patch_embed(unet_out)
        encoded = self.transformer_encoder(tokens)
        pooled = encoded.mean(dim=1)
        V = self.final_linear(pooled).view(-1)
        V = V * self.output_scale
        if self.energy_clamp is not None: V = soft_clamp(V, self.energy_clamp)
        return V

    def velocity(self, x, t):
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)
            V = self.potential(x, t)
            dVdx = torch.autograd.grad(outputs=V, inputs=x, grad_outputs=torch.ones_like(V), create_graph=True)[0]
            return -dVdx

    def forward(self, t, x, return_potential=False, *args, **kwargs):
        return self.velocity(x, t) if not return_potential else self.potential(x, t)


# --- Ablation Model 2: Convolutional Head (No ViT) ---

class ConvHead_EBMWrapper(UNetModelWrapper):
    """
    This model uses a simple Convolutional Head instead of a ViT.
    """
    def __init__(self, **kwargs):
        # Filter arguments for the parent class
        unet_kwargs = filter_kwargs(UNetModelWrapper.__init__, kwargs)
        super().__init__(**unet_kwargs)
        
        dim = kwargs['dim']
        self.output_scale = kwargs.get('output_scale', 1000.0)
        self.energy_clamp = kwargs.get('energy_clamp')

        # The U-Net will output dim[0] channels. The conv_head must accept this.
        self.conv_head = nn.Conv2d(dim[0], 1, kernel_size=1)

    def potential(self, x, t, **kwargs):
        t_dummy = dummy_time(x, value=0.5)
        unet_out = super().forward(t_dummy, x)
        dense_energy_map = self.conv_head(unet_out)
        V = dense_energy_map.mean(dim=[1, 2, 3])
        V = V * self.output_scale
        if self.energy_clamp is not None: V = soft_clamp(V, self.energy_clamp)
        return V

    def velocity(self, x, t):
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)
            V = self.potential(x, t)
            dVdx = torch.autograd.grad(outputs=V, inputs=x, grad_outputs=torch.ones_like(V), create_graph=True)[0]
            return -dVdx

    def forward(self, t, x, return_potential=False, *args, **kwargs):
        return self.velocity(x, t) if not return_potential else self.potential(x, t)

def soft_clamp(x, clamp_val):
    return clamp_val * torch.tanh(x / clamp_val)