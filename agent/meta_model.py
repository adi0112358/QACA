import torch
import torch.nn as nn


class MetaStateModel(nn.Module):

    def __init__(self, state_size=64):
        super().__init__()

        # state + prediction_error + uncertainty + state_energy
        input_size = state_size + 3

        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    # ---------------------------------

    def forward(self, state, prediction_error, uncertainty):

        state = state.view(state.size(0), -1)

        # state energy
        state_energy = torch.norm(state, dim=1, keepdim=True)

        error_tensor = torch.tensor(
            [[prediction_error]],
            dtype=torch.float32,
            device=state.device
        )

        uncertainty_tensor = torch.tensor(
            [[uncertainty]],
            dtype=torch.float32,
            device=state.device
        )

        x = torch.cat(
            [state, error_tensor, uncertainty_tensor, state_energy],
            dim=1
        )

        return self.model(x)