import torch
import math

def get_warmup_lr_lambda(warmup_steps):
    """
    Returns a lambda function for the learning rate scheduler that performs a linear warmup.
    """
    def lr_lambda(step):
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)
    return lr_lambda

def sde_epsilon(t: float, at_data_mask: torch.Tensor, epsilon_max: float, time_cutoff: float) -> torch.Tensor:
    """
    Returns a tensor of shape (batch_size,) with epsilon values for each sample
    at the given time t. If at_data_mask[i] == True, that sample uses epsilon_max.
    Otherwise, it follows a piecewise schedule.
    """
    e_val = torch.zeros_like(at_data_mask, dtype=torch.float32)
    e_val = e_val.to(at_data_mask.device)

    not_data_mask = ~at_data_mask

    if t < time_cutoff:
        e_val[not_data_mask] = 0.0
    elif t < 1.0:
        frac = (t - time_cutoff) / (1.0 - time_cutoff)
        e_val[not_data_mask] = frac * epsilon_max
    else:
        e_val[not_data_mask] = epsilon_max

    e_val[at_data_mask] = epsilon_max
    return e_val

def gibbs_sampling_time_sweep(
    x_init: torch.Tensor,
    model,
    at_data_mask: torch.Tensor,
    n_steps: int,
    dt: float,
    epsilon_max: float,
    time_cutoff: float,
):
    """
    Performs MALA sampling from t=0 to t=(n_steps*dt) with a time-dependent temperature.
    """
    device = x_init.device
    samples = x_init.clone().detach().to(device)
    at_data_mask = at_data_mask.to(device=device, dtype=torch.bool)

    batch_size = samples.shape[0]

    for i in range(n_steps):
        t_val = i * dt
        e_val = sde_epsilon(t_val, at_data_mask, epsilon_max, time_cutoff)
        noise_std = torch.sqrt(2.0 * dt * e_val)

        samples.requires_grad_(True)
        t_tensor = torch.full((batch_size,), t_val, device=samples.device, dtype=samples.dtype)

        V = model.potential(samples, t_tensor)
        grad_V = torch.autograd.grad(V.sum(), samples, create_graph=False)[0]

        with torch.no_grad():
            noise = torch.randn_like(samples) * noise_std.view(-1, *([1]*(samples.ndim-1)))
            samples = samples - dt * grad_V + noise

    return samples.detach()

def flow_weight(t, cutoff=0.8):
    """
    Flow weighting function.
    """
    w = torch.ones_like(t)
    decay_region = (t >= cutoff) & (t < 1.0)
    if torch.any(decay_region):
        w[decay_region] = 1.0 - (t[decay_region] - cutoff) / (1.0 - cutoff)
    w[t >= 1.0] = 0.0
    return w

def ema(source, target, decay):
    """
    Exponential Moving Average update.
    """
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )

def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y
