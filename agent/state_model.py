import torch
import torch.nn as nn


class StateModel(nn.Module):

    def __init__(self, obs_size=100, action_size=4, state_size=64):

        super().__init__()

        grid_size = int(obs_size ** 0.5)

        self.grid_size = grid_size
        self.state_size = state_size

        # --------------------------------
        # CNN Encoder
        # --------------------------------

        self.encoder = nn.Sequential(

            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(32 * grid_size * grid_size, 128),
            nn.ReLU(),

            nn.Linear(128, 64)
        )

        # --------------------------------
        # GRU Memory
        # --------------------------------

        self.rnn = nn.GRU(
            input_size=64 + action_size,
            hidden_size=state_size,
            batch_first=True
        )

    # --------------------------------

    def forward(self, obs, action, state):

        batch = obs.shape[0]

        grid = obs.view(batch, 1, self.grid_size, self.grid_size)

        z = self.encoder(grid)

        x = torch.cat([z, action], dim=1).unsqueeze(1)

        output, new_state = self.rnn(x, state)

        return new_state

    # --------------------------------

    def init_state(self):

        return torch.zeros(1, 1, self.state_size)