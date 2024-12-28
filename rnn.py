import torch
import torch.nn as nn


# Pytorch implementation
class RNNCell(nn.Module):
    """Initializes a simple RNN cell."""

    def __init__(self, input_size, hidden_size, output_size):
        super(RNNCell, self).__init__()
        self.hidden_size = hidden_size

        # Replace individual weight matrices with Linear layers
        self.xh = nn.Linear(input_size, hidden_size)  # Combines Wxh and bh
        self.hh = nn.Linear(hidden_size, hidden_size)  # Combines Whh and bh
        self.hy = nn.Linear(hidden_size, output_size)  # Combines Why and by

    def forward(self, x_t, h_prev):
        """Performs forward pass for a single time step.

        Args:
            x_t: Input vector at time t (shape: [batch_size, input_size]).
            h_prev: Hidden state from previous time step (shape: [batch_size, hidden_size]).

        Returns:
            tuple: Contains:
                h_t: New hidden state (shape: [batch_size, hidden_size])
                y_t: Output vector (shape: [batch_size, output_size])
        """
        # Compute hidden state
        h_t = torch.tanh(self.xh(x_t) + self.hh(h_prev))

        # Compute output probabilities
        y_t = self.hy(h_t)

        # Return outputs in original shape [features, batch_size]
        return h_t, y_t

    def init_weights(self):
        """Initializes the weights of all linear layers.

        Uses normal distribution (mean=0, std=0.01) for weights and zeros for biases.
        """
        for layer in [self.xh, self.hh, self.hy]:
            nn.init.normal_(
                layer.weight, mean=0.0, std=0.01
            )  # Standard normal with std = 0.01
            nn.init.zeros_(layer.bias)  # Initialize biases to zero

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden_size)
