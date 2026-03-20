from dataclasses import dataclass
import torch


@dataclass
class DivergenceConfig:
    w_error: float = 1.0
    w_uncertainty: float = 0.6
    w_value: float = 0.3
    w_reference: float = 0.4
    beta: float = 5.0

    target_error: float = 0.2
    target_uncertainty: float = 1.0
    target_value: float = 0.0
    target_state_norm: float = 3.0


@dataclass
class BarrierConfig:
    max_state_norm: float = 8.0
    max_uncertainty: float = 2.5
    max_error: float = 3.0


def _to_tensor(value, device, dtype):

    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)

    return torch.tensor(value, device=device, dtype=dtype)


def _soft_penalty(x, beta):
    # Smooth penalty that approximates relu(x)
    return torch.nn.functional.softplus(beta * x) / beta


def compute_divergence(state, prediction_error, uncertainty, value, config: DivergenceConfig):

    state_flat = state.view(state.size(0), -1)
    state_norm = torch.norm(state_flat, dim=1, keepdim=True)

    device = state.device
    dtype = state.dtype

    err = _to_tensor(prediction_error, device, dtype).view(1, 1)
    unc = _to_tensor(uncertainty, device, dtype).view(1, 1)
    val = _to_tensor(value, device, dtype).view(1, 1)

    error_term = _soft_penalty(err - config.target_error, config.beta)
    unc_term = _soft_penalty(unc - config.target_uncertainty, config.beta)
    value_term = _soft_penalty(config.target_value - val, config.beta)

    ref_term = (state_norm - config.target_state_norm) ** 2

    divergence = (
        config.w_error * error_term
        + config.w_uncertainty * unc_term
        + config.w_value * value_term
        + config.w_reference * ref_term
    )

    return divergence.mean().item()


def compute_barriers(state, prediction_error, uncertainty, config: BarrierConfig):

    state_flat = state.view(state.size(0), -1)
    state_norm = torch.norm(state_flat, dim=1, keepdim=True).mean().item()

    barriers = {
        "state_norm": config.max_state_norm - state_norm,
        "uncertainty": config.max_uncertainty - float(uncertainty),
        "prediction_error": config.max_error - float(prediction_error),
    }

    return barriers
