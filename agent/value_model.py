import torch
import torch.nn as nn


class ValueModel(nn.Module):

    def __init__(self, state_size=64):
        super().__init__()

        self.model = nn.Sequential(

            nn.LayerNorm(state_size),

            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    # ---------------------------------

    def forward(self, state):

        state = state.view(state.size(0), -1)

        value = self.model(state)

        return value