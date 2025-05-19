import torch
import math
from torch import nn


class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 32):
        """Creates a simple MLP in torch with 2 hidden layers of size `hidden_state`

        Args:
            in_dim (int): input dimension
            out_dim (int): output dimensions
            hidden_dim (int, optional): size of hidden layers. Defaults to 32.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
