import torch
import torch.nn as nn


class ValueTrainer:

    def __init__(self, value_model, lr=1e-3, gamma=0.99):

        self.value_model = value_model
        self.gamma = gamma

        self.optimizer = torch.optim.Adam(value_model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    # --------------------------------------------------

    def train_step(self, state, reward, next_state):

        # ensure batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        if len(next_state.shape) == 1:
            next_state = next_state.unsqueeze(0)

        # detach states from previous graphs
        state = state.detach()
        next_state = next_state.detach()

        # predicted value
        value = self.value_model(state)

        # next state value (no gradient)
        with torch.no_grad():
            next_value = self.value_model(next_state)

        # TD target
        target = reward + self.gamma * next_value

        # loss
        loss = self.loss_fn(value, target)

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()