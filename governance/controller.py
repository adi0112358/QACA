from dataclasses import dataclass
from typing import Optional
import numpy as np

from .qp_solver import QPSolverConfig, solve_clf_cbf_qp


@dataclass
class GovernanceControllerConfig:
    u_min: float = 0.0
    u_max: float = 1.0
    u_ref: float = 0.2

    clf_eta: float = 0.1
    k_v: float = 0.6
    k_b: float = 1.0

    u_weight: float = 1.0
    slack_weight: float = 20.0

    min_horizon: int = 2
    max_horizon: int = 6
    min_samples: int = 32
    max_samples: int = 128

    min_uncertainty_weight: float = 0.05
    max_uncertainty_weight: float = 0.3

    safe_mode_floor: float = 0.9


@dataclass
class GovernanceAction:
    u: float
    exploration_scale: float
    horizon: int
    num_samples: int
    uncertainty_weight: float
    safe_mode: bool
    slack: float
    active_barrier: Optional[str]


class GovernanceController:

    def __init__(self, config: GovernanceControllerConfig):
        self.config = config

    # --------------------------------------------------

    # --------------------------------------------------

    def compute_action(self, divergence, barriers, budget_scale):

        cfg = self.config
        u, slack, diag = solve_clf_cbf_qp(
            divergence,
            barriers,
            QPSolverConfig(
                u_min=cfg.u_min,
                u_max=cfg.u_max,
                u_ref=cfg.u_ref,
                k_v=cfg.k_v,
                k_b=cfg.k_b,
                clf_eta=cfg.clf_eta,
                slack_weight=cfg.slack_weight,
                u_weight=cfg.u_weight
            )
        )

        safe_mode = (u >= cfg.safe_mode_floor and divergence > 0) or (not diag.get("feasible", True))

        effective_scale = np.clip(0.5 * budget_scale + 0.5 * u, 0.0, 1.0)

        horizon = int(round(cfg.min_horizon + effective_scale * (cfg.max_horizon - cfg.min_horizon)))
        num_samples = int(round(cfg.min_samples + effective_scale * (cfg.max_samples - cfg.min_samples)))

        uncertainty_weight = (
            cfg.min_uncertainty_weight
            + u * (cfg.max_uncertainty_weight - cfg.min_uncertainty_weight)
        )

        exploration_scale = max(0.1, 1.0 - 0.7 * u)

        if safe_mode:
            horizon = cfg.min_horizon
            num_samples = cfg.min_samples
            uncertainty_weight = cfg.max_uncertainty_weight
            exploration_scale = 0.1

        return GovernanceAction(
            u=u,
            exploration_scale=exploration_scale,
            horizon=horizon,
            num_samples=num_samples,
            uncertainty_weight=uncertainty_weight,
            safe_mode=safe_mode,
            slack=slack,
            active_barrier=diag.get("active_barrier")
        )
