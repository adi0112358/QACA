from dataclasses import dataclass
import torch


@dataclass
class GovernanceConfig:
    max_state_norm: float = 10.0
    max_state_delta: float = 4.0
    max_uncertainty: float = 2.5
    lyapunov_decay: float = 0.05

    w_barrier: float = 1.0
    w_lyapunov: float = 0.6
    w_uncertainty: float = 0.5
    w_delta: float = 0.3


def _state_norm(state):
    state_flat = state.view(state.size(0), -1)
    return torch.norm(state_flat, dim=1, keepdim=True)


def governance_penalty(state, next_state, logvar, config: GovernanceConfig):
    """
    Control-inspired governance penalty:
    - Lyapunov-style decay: discourage growth in state energy
    - Barrier on latent norm
    - Barrier on large step sizes (delta)
    - Uncertainty barrier from world model variance
    """

    prev_norm = _state_norm(state)
    next_norm = _state_norm(next_state)

    # Lyapunov-style condition: V(next) - V(prev) <= -c * V(prev)
    prev_energy = prev_norm ** 2
    next_energy = next_norm ** 2
    lyapunov_delta = next_energy - prev_energy + config.lyapunov_decay * prev_energy
    penalty_lyapunov = torch.relu(lyapunov_delta)

    # Barrier on latent norm
    penalty_barrier = torch.relu(next_norm - config.max_state_norm)

    # Barrier on large latent jumps
    delta = torch.norm((next_state - state).view(state.size(0), -1), dim=1, keepdim=True)
    penalty_delta = torch.relu(delta - config.max_state_delta)

    # Barrier on uncertainty
    uncertainty = torch.mean(torch.exp(logvar), dim=1, keepdim=True)
    penalty_uncertainty = torch.relu(uncertainty - config.max_uncertainty)

    penalty = (
        config.w_lyapunov * penalty_lyapunov
        + config.w_barrier * penalty_barrier
        + config.w_delta * penalty_delta
        + config.w_uncertainty * penalty_uncertainty
    )

    return torch.mean(penalty)
