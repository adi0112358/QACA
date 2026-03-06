import torch
import torch.nn as nn


class StateModel(nn.Module):
    def __init__(self, obs_size=25, action_size=4, state_size=64):
        super().__init__()

        self.state_size = state_size

        self.rnn = nn.GRU(
            input_size=obs_size + action_size,
            hidden_size=state_size,
            batch_first=True
        )

    def forward(self, obs, action, state):

        x = torch.cat([obs, action], dim=-1).unsqueeze(0)

        output, new_state = self.rnn(x, state)

        return new_state

    def init_state(self):
        return torch.zeros(1, 1, self.state_size)