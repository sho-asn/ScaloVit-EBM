import torch
from absl import flags

# This is a placeholder since we are not using absl flags directly in this script,
# but the functions were written to use it. We can define dummy flags.
class DummyFlags:
    def __init__(self):
        self.epsilon_max = 0.1
        self.time_cutoff = 0.9

FLAGS = DummyFlags()

def sde_epsilon(t: float, at_data_mask: torch.Tensor) -> torch.Tensor:
    """
    Returns a tensor of shape (batch_size,) with epsilon values for each sample
    at the given time t. If at_data_mask[i] == True, that sample uses epsilon_max.
    Otherwise, it follows a piecewise schedule.
    """
    eps_max = FLAGS.epsilon_max
    cutoff = FLAGS.time_cutoff

    e_val = torch.zeros_like(at_data_mask, dtype=torch.float32)
    e_val = e_val.to(at_data_mask.device)

    not_data_mask = ~at_data_mask

    if t < cutoff:
        e_val[not_data_mask] = 0.0
    elif t < 1.0:
        frac = (t - cutoff) / (1.0 - cutoff)
        e_val[not_data_mask] = frac * eps_max
    else:
        e_val[not_data_mask] = eps_max

    e_val[at_data_mask] = eps_max
    return e_val

def gibbs_sampling_time_sweep(
    x_init: torch.Tensor,
    model,
    at_data_mask: torch.Tensor,
    n_steps: int = 150,
    dt: float = 0.01,
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
        e_val = sde_epsilon(t_val, at_data_mask)
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
            yield x
