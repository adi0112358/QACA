import torch
import numpy as np


class MPCPlanner:

    def __init__(
        self,
        world_model,
        value_model,
        horizon=3,
        num_samples=8,
        uncertainty_weight=0.1
    ):

        self.world_model = world_model
        self.value_model = value_model

        self.horizon = horizon
        self.num_samples = num_samples

        self.uncertainty_weight = uncertainty_weight


    # --------------------------------------------------

    def simulate_trajectory(self, state, actions):

        current_state = state
        total_cost = 0

        for action in actions:

            action_vec = torch.zeros(1, 4)
            action_vec[0, action] = 1

            mu, logvar = self.world_model(current_state, action_vec)

            next_state = mu

            uncertainty = torch.mean(torch.exp(logvar))

            value = self.value_model(next_state)

            cost = -value + self.uncertainty_weight * uncertainty

            total_cost += cost.item()

            current_state = next_state

        return total_cost


    # --------------------------------------------------

    def sample_action_sequence(self):

        return np.random.randint(0, 4, size=self.horizon)


    # --------------------------------------------------

    def select_action(self, state):

        best_cost = float("inf")
        best_sequence = None

        for _ in range(self.num_samples):

            actions = self.sample_action_sequence()

            cost = self.simulate_trajectory(state, actions)

            if cost < best_cost:
                best_cost = cost
                best_sequence = actions

        return int(best_sequence[0])