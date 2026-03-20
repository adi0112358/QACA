from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class QPSolverConfig:
    u_min: float
    u_max: float
    u_ref: float
    k_v: float
    k_b: float
    clf_eta: float
    slack_weight: float
    u_weight: float = 1.0


def solve_clf_cbf_qp(
    divergence: float,
    barriers: Dict[str, float],
    config: QPSolverConfig
) -> Tuple[float, float, Dict[str, float]]:
    """
    Solve a 1D CLF/CBF QP with slack:
        minimize (u - u_ref)^2 + rho * s^2
        s.t. u_min <= u <= u_max
             k_b * u >= -B_j  for all barriers
             -k_v * u <= -clf_eta * V + s
             s >= 0
    Returns (u, slack, diagnostics)
    """

    # Barrier lower bound
    lower = config.u_min
    active = {}

    for name, B in barriers.items():
        if B < 0:
            req = -B / max(config.k_b, 1e-6)
            if req > lower:
                lower = req
                active["barrier"] = name

    lower = max(lower, config.u_min)
    upper = config.u_max

    if lower > upper:
        # infeasible, clamp to max
        return upper, max(0.0, config.clf_eta * divergence - config.k_v * upper), {
            "feasible": False,
            "active_barrier": active.get("barrier")
        }

    # CLF zero-slack boundary
    if config.k_v > 1e-6:
        u_clf = config.clf_eta * divergence / config.k_v
    else:
        u_clf = lower

    # Region 1: slack = 0, u >= u_clf
    if config.u_ref >= u_clf:
        u_star = min(max(config.u_ref, lower), upper)
        if u_star >= u_clf:
            return u_star, 0.0, {"feasible": True, "active_barrier": active.get("barrier")}

    # Region 2: slack > 0, u < u_clf
    denom = config.u_weight + config.slack_weight * (config.k_v ** 2)
    if denom <= 1e-9:
        denom = 1e-9
    u_unclamped = (
        config.u_weight * config.u_ref
        + config.slack_weight * config.k_v * config.clf_eta * divergence
    ) / denom

    u_star = min(max(u_unclamped, lower), upper)
    u_star = min(u_star, u_clf)

    slack = max(0.0, config.clf_eta * divergence - config.k_v * u_star)

    return u_star, slack, {"feasible": True, "active_barrier": active.get("barrier")}
