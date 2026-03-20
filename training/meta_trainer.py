import torch
import torch.nn as nn


class MetaTrainer:

    def __init__(self, meta_model, lr=1e-3, uncertainty_center=1.0):

        self.meta_model = meta_model
        self.optimizer = torch.optim.Adam(meta_model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.uncertainty_center = uncertainty_center

    # --------------------------------------------------

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

    # --------------------------------------------------

    def compute_targets(self, state, next_state, prediction_error, uncertainty):

        batch = state.size(0)
        device = state.device
        dtype = state.dtype

        error = self._to_batch_tensor(prediction_error, batch, device, dtype)
        unc = self._to_batch_tensor(uncertainty, batch, device, dtype)

        delta = torch.norm((next_state - state).view(batch, -1), dim=1, keepdim=True)

        # Heuristic meta-targets (0..1-ish ranges)
        reliability = torch.exp(-error)
        stability = torch.exp(-delta)
        risk = torch.sigmoid(unc - self.uncertainty_center)
        novelty = torch.tanh(error)

        targets = torch.cat([reliability, stability, risk, novelty], dim=1)

        return targets

    # --------------------------------------------------

    def train_step(self, state, next_state, prediction_error, uncertainty):

        if len(state.shape) == 2:
            state = state.unsqueeze(0)

        if len(next_state.shape) == 2:
            next_state = next_state.unsqueeze(0)

        state = state.detach()
        next_state = next_state.detach()

        targets = self.compute_targets(state, next_state, prediction_error, uncertainty)

        preds = self.meta_model(state, prediction_error, uncertainty)

        loss = self.loss_fn(preds, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
