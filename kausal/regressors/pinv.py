import torch

from .base import BaseRegressor

class PINV(BaseRegressor):
    """
    Estimate Koopman Operator with MoorePenroseInverse method
        Reference: https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, 
        W = None, 
        Wt = None
    ):
        if W is None or Wt is None:
            raise ValueError("Observables at both current and shifted step must be provided.")
            
        W_inverse = torch.linalg.pinv(W)
        K = Wt @ W_inverse
        return K
        