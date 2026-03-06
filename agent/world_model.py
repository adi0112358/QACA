import torch
import torch.nn as nn


class WorldModel(nn.Module):

    def __init__(self, state_size=64, action_size=4):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_size + action_size, 128),
            nn.ReLU(),
            nn.Linear(128, state_size)
        )

    def forward(self, state, action):

        # flatten batch if needed
        state = state.view(state.size(0), -1)
        action = action.view(action.size(0), -1)

        x = torch.cat([state, action], dim=1)

        next_state_pred = self.model(x)

        return next_state_pred