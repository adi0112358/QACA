import numpy as np
import torch
from typing import Optional

from governance.constraints import governance_penalty, GovernanceConfig


class CEMPlanner:

    def __init__(
        self,
        world_model,
        value_model,
        horizon=4,
        num_samples=64,
        elite_frac=0.2,
        iterations=3,
        alpha=0.7,
        uncertainty_weight=0.1,
        governance_config: Optional[GovernanceConfig] = None,
        min_horizon=2,
        max_horizon=6,
        min_samples=32,
        max_samples=128
    ):

        self.world_model = world_model
        self.value_model = value_model

        self.horizon = horizon
        self.num_samples = num_samples
        self.elite_frac = elite_frac
        self.iterations = iterations
        self.alpha = alpha
        self.uncertainty_weight = uncertainty_weight

        self.action_size = 4
        self.governance_config = governance_config

        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.min_samples = min_samples
        self.max_samples = max_samples


    # --------------------------------------------------

    def set_budget(self, horizon=None, num_samples=None):

        if horizon is not None:
            self.horizon = int(np.clip(horizon, self.min_horizon, self.max_horizon))

        if num_samples is not None:
            self.num_samples = int(np.clip(num_samples, self.min_samples, self.max_samples))

    # --------------------------------------------------

    def set_uncertainty_weight(self, weight):

        self.uncertainty_weight = float(weight)


    # --------------------------------------------------

    def _sample_sequences(self, probs):

        samples = np.zeros((self.num_samples, self.horizon), dtype=np.int64)

        for t in range(self.horizon):
            samples[:, t] = np.random.choice(
                self.action_size,
                size=self.num_samples,
                p=probs[t]
            )

        return samples


    # --------------------------------------------------

    def simulate_trajectory(self, state, actions):

        current_state = state
        total_cost = 0.0

        for action in actions:

            action_vec = torch.zeros(1, self.action_size)
            action_vec[0, action] = 1

            mu, logvar = self.world_model(current_state, action_vec)
            next_state = mu

            uncertainty = torch.mean(torch.exp(logvar))
            value = self.value_model(next_state)

            cost = -value + self.uncertainty_weight * uncertainty

            if self.governance_config is not None:
                cost = cost + governance_penalty(
                    current_state,
                    next_state,
                    logvar,
                    self.governance_config
                )

            total_cost += cost.item()
            current_state = next_state

        return total_cost


    # --------------------------------------------------

    def select_action(self, state):

        probs = np.full((self.horizon, self.action_size), 1.0 / self.action_size)
        best_sequence = None
        best_cost = float("inf")

        for _ in range(self.iterations):

            sequences = self._sample_sequences(probs)

            costs = np.zeros(self.num_samples)

            for i in range(self.num_samples):
                costs[i] = self.simulate_trajectory(state, sequences[i])

            elite_count = max(1, int(self.elite_frac * self.num_samples))
            elite_idx = np.argsort(costs)[:elite_count]
            elites = sequences[elite_idx]

            new_probs = np.zeros_like(probs)
            for t in range(self.horizon):
                for a in range(self.action_size):
                    new_probs[t, a] = np.mean(elites[:, t] == a)

            probs = (1 - self.alpha) * probs + self.alpha * new_probs
            probs = probs / probs.sum(axis=1, keepdims=True)

            if costs[elite_idx[0]] < best_cost:
                best_cost = costs[elite_idx[0]]
                best_sequence = elites[0]

        return int(best_sequence[0])
