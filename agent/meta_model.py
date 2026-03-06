import torch
import torch.nn as nn


class MetaStateModel(nn.Module):

    def __init__(self, state_size=64):

        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_size + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, state, prediction_error):

        state = state.view(state.size(0), -1)

        error_tensor = torch.tensor([[prediction_error]], dtype=torch.float32)

        x = torch.cat([state, error_tensor], dim=1)

        return self.model(x)