import torch
import torch.nn as nn
import copy


class ValueTrainer:

    def __init__(self, value_model, lr=1e-3, gamma=0.99, target_update=100):

        self.value_model = value_model
        self.target_model = copy.deepcopy(value_model)

        self.gamma = gamma
        self.target_update = target_update

        self.optimizer = torch.optim.Adam(value_model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.step_count = 0

    # --------------------------------------------------

    def train_step(self, state, reward, next_state):

        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        if len(next_state.shape) == 1:
            next_state = next_state.unsqueeze(0)

        state = state.detach()
        next_state = next_state.detach()

        # current value
        value = self.value_model(state)

        # target value (no gradient)
        with torch.no_grad():
            next_value = self.target_model(next_state)

        target = reward + self.gamma * next_value

        loss = self.loss_fn(value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.step_count += 1

        if self.step_count % self.target_update == 0:
            self.target_model.load_state_dict(self.value_model.state_dict())

        return loss.item()