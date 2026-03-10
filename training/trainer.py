import torch


class WorldModelTrainer:

    def __init__(self, model, lr=1e-3):

        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---------------------------------

    def gaussian_nll(self, target, mu, logvar):
        """
        Negative log likelihood for Gaussian prediction
        """

        var = torch.exp(logvar)

        loss = ((target - mu) ** 2) / (var + 1e-6) + logvar

        return torch.mean(loss)

    # ---------------------------------

    def train_step(self, states, actions, next_states):

        states = torch.cat(states)
        actions = torch.cat(actions)
        next_states = torch.cat(next_states)

        states = states.squeeze(1)
        next_states = next_states.squeeze(1)

        # forward pass
        mu, logvar = self.model(states, actions)

        # probabilistic loss
        loss = self.gaussian_nll(next_states, mu, logvar)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()