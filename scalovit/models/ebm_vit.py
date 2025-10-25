"""Vision Transformer-based energy model wrappers."""

# Adapted from EnergyMatching/experiments/cifar10/network_transformer_vit.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchcfm.models.unet.unet import UNetModelWrapper


class PatchEmbed(nn.Module):
    """Split (B, C, H, W) feature maps into per-patch embeddings."""

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int | tuple[int, int] = 4,
        embed_dim: int = 128,
        image_size: tuple[int, int] = (32, 32),
        include_pos_embed: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches_h = image_size[0] // self.patch_size[0]
        self.num_patches_w = image_size[1] // self.patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.include_pos_embed = include_pos_embed
        if include_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        else:
            self.pos_embed = None

        nn.init.xavier_uniform_(self.patch_embed.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        batch, embed_dim, h_patches, w_patches = x.shape
        x = x.view(batch, embed_dim, h_patches * w_patches).transpose(1, 2)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        return x


def soft_clamp(x: torch.Tensor, clamp_val: float) -> torch.Tensor:
    """Tanh-based clamp returning values in [-clamp_val, clamp_val]."""

    return clamp_val * torch.tanh(x / clamp_val)


def dummy_time(x: torch.Tensor, value: float = 0.5) -> torch.Tensor:
    """Create a (B,) tensor filled with ``value`` on x's device."""

    return torch.full((x.shape[0],), value, device=x.device, dtype=x.dtype)


class EBViTModelWrapper(UNetModelWrapper):
    """Energy model with a ViT head providing per-patch energies."""

    def __init__(
        self,
        dim: tuple[int, int, int] = (3, 32, 32),
        num_channels: int = 128,
        num_res_blocks: int = 2,
        channel_mult: tuple[int, ...] = (1, 2, 2, 2),
        attention_resolutions: str = "16",
        num_heads: int = 4,
        num_head_channels: int = 64,
        dropout: float = 0.1,
        class_cond: bool = False,
        learn_sigma: bool = False,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        resblock_updown: bool = False,
        use_scale_shift_norm: bool = False,
        use_new_attention_order: bool = False,
        patch_size: int | tuple[int, int] = 4,
        embed_dim: int = 128,
        transformer_nheads: int = 4,
        transformer_nlayers: int = 2,
        include_pos_embed: bool = True,
        output_scale: float = 1000.0,
        energy_clamp: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            dim=dim,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            class_cond=class_cond,
            learn_sigma=learn_sigma,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            resblock_updown=resblock_updown,
            use_scale_shift_norm=use_scale_shift_norm,
            use_new_attention_order=use_new_attention_order,
            **kwargs,
        )

        self.out_channels = dim[0]
        self.output_scale = output_scale
        self.energy_clamp = energy_clamp

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
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_nlayers,
        )

        self.final_linear = nn.Linear(embed_dim, 1)

    def potential(self, x: torch.Tensor, t: torch.Tensor, return_tokens: bool = False):
        t_dummy = dummy_time(x, value=0.5)
        unet_out = super().forward(t_dummy, x)
        tokens = self.patch_embed(unet_out)
        encoded = self.transformer_encoder(tokens)
        energies = self.final_linear(encoded).squeeze(-1) * self.output_scale
        if self.energy_clamp is not None:
            energies = soft_clamp(energies, self.energy_clamp)
        if return_tokens:
            return energies, encoded
        return energies

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)
            energies = self.potential(x, t)
            total_energy = energies.sum()
            grad = torch.autograd.grad(total_energy, x, create_graph=True)[0]
            return -grad

    def forward(self, t: torch.Tensor, x: torch.Tensor, return_potential: bool = False, *args, **kwargs):
        if return_potential:
            return self.potential(x, t)
        return self.velocity(x, t)
