"""Core helpers for energy-based model training."""

from __future__ import annotations

from typing import Callable

import torch


def get_warmup_lr_lambda(warmup_steps: int) -> Callable[[int], float]:
    """Return a linear warmup schedule lambda for the optimiser."""

    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    return lr_lambda


def sde_epsilon(
    t: float,
    at_data_mask: torch.Tensor,
    epsilon_max: float,
    time_cutoff: float,
) -> torch.Tensor:
    """Piecewise epsilon schedule with optional data mask."""

    epsilons = torch.zeros_like(at_data_mask, dtype=torch.float32, device=at_data_mask.device)
    not_data_mask = ~at_data_mask

    if t < time_cutoff:
        epsilons[not_data_mask] = 0.0
    elif t < 1.0:
        frac = (t - time_cutoff) / (1.0 - time_cutoff)
        epsilons[not_data_mask] = frac * epsilon_max
    else:
        epsilons[not_data_mask] = epsilon_max

    epsilons[at_data_mask] = epsilon_max
    return epsilons


def gibbs_sampling_time_sweep(
    x_init: torch.Tensor,
    model,
    at_data_mask: torch.Tensor,
    n_steps: int,
    dt: float,
    epsilon_max: float,
    time_cutoff: float,
) -> torch.Tensor:
    """Run Langevin-like sampling with a time-dependent temperature."""

    device = x_init.device
    samples = x_init.clone().detach().to(device)
    at_data_mask = at_data_mask.to(device=device, dtype=torch.bool)
    batch_size = samples.shape[0]

    for i in range(n_steps):
        t_val = i * dt
        eps = sde_epsilon(t_val, at_data_mask, epsilon_max, time_cutoff)
        noise_std = torch.sqrt(2.0 * dt * eps)

        samples.requires_grad_(True)
        t_tensor = torch.full((batch_size,), t_val, device=samples.device, dtype=samples.dtype)
        energies = model.potential(samples, t_tensor)
        grad = torch.autograd.grad(energies.sum(), samples, create_graph=False)[0]

        with torch.no_grad():
            noise = torch.randn_like(samples) * noise_std.view(-1, *([1] * (samples.ndim - 1)))
            samples = samples - dt * grad + noise

    return samples.detach()


def flow_weight(t: torch.Tensor, cutoff: float = 0.8) -> torch.Tensor:
    """Return weights for flow-matching loss with linear decay beyond cutoff."""

    weights = torch.ones_like(t)
    decay_region = (t >= cutoff) & (t < 1.0)
    if torch.any(decay_region):
        weights[decay_region] = 1.0 - (t[decay_region] - cutoff) / (1.0 - cutoff)
    weights[t >= 1.0] = 0.0
    return weights


def ema(source: torch.nn.Module, target: torch.nn.Module, decay: float) -> None:
    """In-place exponential moving average update of target params."""

    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay + source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    """Yield batches indefinitely from a standard dataloader."""

    while True:
        for batch in iter(dataloader):
            yield batch
