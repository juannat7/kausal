import torch
from .base import BaseObservables

class IdentityFeatures(BaseObservables):
    """
    Returns itself, I(w) = w.

    Parameters:
        None

    Returns
        Identity (torch.Tensor): The Identity transforms.
    """

    def __init__(self):
        super().__init__()

    def fit(self, x = None, y = None, **kwargs):
        return self
    
    def forward(
        self, 
        X
    ):
        return X
