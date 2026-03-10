import torch
import torch.nn as nn


class WorldModel(nn.Module):

    def __init__(self, state_size=64, action_size=4):
        super().__init__()

        self.state_size = state_size

        self.model = nn.Sequential(
            nn.Linear(state_size + action_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # output layers
        self.mu_head = nn.Linear(128, state_size)
        self.logvar_head = nn.Linear(128, state_size)

    # ---------------------------------

    def forward(self, state, action):

        state = state.view(state.size(0), -1)
        action = action.view(action.size(0), -1)

        x = torch.cat([state, action], dim=1)

        h = self.model(x)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        return mu, logvar

    # ---------------------------------

    def sample_next_state(self, mu, logvar):

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return mu + eps * std