import torch
import torch.nn as nn


class WorldModelTrainer:

    def __init__(self, model, lr=1e-3):

        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, states, actions, next_states):

        states = torch.cat(states)
        actions = torch.cat(actions)
        next_states = torch.cat(next_states)

        states = states.squeeze(1)
        next_states = next_states.squeeze(1)

        predicted = self.model(states, actions)

        loss = self.loss_fn(predicted, next_states)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()