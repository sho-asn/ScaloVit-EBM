"""Alternative energy heads used for ablation studies."""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn

from torchcfm.models.unet.unet import UNetModelWrapper

from .ebm_vit import PatchEmbed, dummy_time, soft_clamp


def filter_kwargs(func, kwargs):
    """Filter keyword arguments so only parameters accepted by ``func`` remain."""

    signature = inspect.signature(func)
    allowed = {
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in kwargs.items() if key in allowed}


class ImageBased_EBViTModelWrapper(UNetModelWrapper):
    """Produce a single energy score per chunk using a transformer head."""

    def __init__(self, **kwargs):
        unet_kwargs = filter_kwargs(UNetModelWrapper.__init__, kwargs)
        super().__init__(**unet_kwargs)

        dim = kwargs["dim"]
        patch_size = kwargs["patch_size"]
        embed_dim = kwargs["embed_dim"]
        transformer_nheads = kwargs["transformer_nheads"]
        transformer_nlayers = kwargs["transformer_nlayers"]
        dropout = kwargs.get("dropout", 0.1)
        include_pos_embed = kwargs.get("include_pos_embed", True)

        self.out_channels = dim[0]
        self.output_scale = kwargs.get("output_scale", 1000.0)
        self.energy_clamp = kwargs.get("energy_clamp")

        self.patch_embed = PatchEmbed(
            in_channels=self.out_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            image_size=dim[1:],
            include_pos_embed=include_pos_embed,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_nheads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_nlayers)
        self.final_linear = nn.Linear(embed_dim, 1)

    def potential(self, x, t, **kwargs):
        t_dummy = dummy_time(x, value=0.5)
        unet_out = super().forward(t_dummy, x)
        tokens = self.patch_embed(unet_out)
        encoded = self.transformer_encoder(tokens)
        pooled = encoded.mean(dim=1)
        energy = self.final_linear(pooled).view(-1) * self.output_scale
        if self.energy_clamp is not None:
            energy = soft_clamp(energy, self.energy_clamp)
        return energy

    def velocity(self, x, t):
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)
            energy = self.potential(x, t)
            grad = torch.autograd.grad(
                outputs=energy,
                inputs=x,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]
            return -grad

    def forward(self, t, x, return_potential=False, *args, **kwargs):
        return self.potential(x, t) if return_potential else self.velocity(x, t)


class ConvHead_EBMWrapper(UNetModelWrapper):
    """Replace the ViT head with a simple 1x1 convolutional head."""

    def __init__(self, **kwargs):
        unet_kwargs = filter_kwargs(UNetModelWrapper.__init__, kwargs)
        super().__init__(**unet_kwargs)

        dim = kwargs["dim"]
        self.output_scale = kwargs.get("output_scale", 1000.0)
        self.energy_clamp = kwargs.get("energy_clamp")

        self.conv_head = nn.Conv2d(dim[0], 1, kernel_size=1)

    def potential(self, x, t, **kwargs):
        t_dummy = dummy_time(x, value=0.5)
        unet_out = super().forward(t_dummy, x)
        dense_map = self.conv_head(unet_out)
        energy = dense_map.mean(dim=[1, 2, 3]) * self.output_scale
        if self.energy_clamp is not None:
            energy = soft_clamp(energy, self.energy_clamp)
        return energy

    def velocity(self, x, t):
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)
            energy = self.potential(x, t)
            grad = torch.autograd.grad(
                outputs=energy,
                inputs=x,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]
            return -grad

    def forward(self, t, x, return_potential=False, *args, **kwargs):
        return self.potential(x, t) if return_potential else self.velocity(x, t)


__all__ = ["ImageBased_EBViTModelWrapper", "ConvHead_EBMWrapper"]
