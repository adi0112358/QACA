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

    def _to_batch_tensor(self, value, batch, device, dtype):

        if torch.is_tensor(value):
            value_tensor = value.to(device=device, dtype=dtype)
            if value_tensor.dim() == 0:
                value_tensor = value_tensor.view(1, 1).repeat(batch, 1)
            elif value_tensor.dim() == 1:
                value_tensor = value_tensor.view(-1, 1)
        else:
            value_tensor = torch.full((batch, 1), float(value), device=device, dtype=dtype)

        return value_tensor

    # ---------------------------------

    def forward(self, state, prediction_error, uncertainty):

        state = state.view(state.size(0), -1)

        # state energy
        state_energy = torch.norm(state, dim=1, keepdim=True)

        batch = state.size(0)

        error_tensor = self._to_batch_tensor(
            prediction_error,
            batch,
            state.device,
            state.dtype
        )

        uncertainty_tensor = self._to_batch_tensor(
            uncertainty,
            batch,
            state.device,
            state.dtype
        )

        x = torch.cat(
            [state, error_tensor, uncertainty_tensor, state_energy],
            dim=1
        )

        return self.model(x)
