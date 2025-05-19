import torch
from torch.nn import functional as F

from .base import BaseRegressor

class NNDMD(BaseRegressor):
    """
    [WIP]: Estimate Koopman Operator with Neural Network based Dynamic Mode Decomposition (NNDMD) method.

    Parameters:
        
    """

    def __init__(
        self,
        init_std = 0.1,
        epochs = 100,
        lr = 1e-2
    ):
        super().__init__()
        self.init_std = init_std
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(
        self, 
        W = None, 
        Wt = None
    ):
        if W is None or Wt is None:
            raise ValueError("Observables at both current and shifted step must be provided.")

        assert W.ndim == 2 and Wt.ndim == 2, "Observables must have a dimension of (n_features, n_timestep)"

        W, Wt = W.to(self.device), Wt.to(self.device)

        # Init K
        K = torch.nn.Parameter(torch.randn(Wt.shape[0], W.shape[0], device=self.device) * self.init_std)

        # Set up optimizer and loss function
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW([K], lr=self.lr)

        # Training loop
        losses = []
        with torch.enable_grad():
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                loss = loss_fn(torch.matmul(K, W), Wt)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        # Get K
        K = K.detach().cpu()
        return K
