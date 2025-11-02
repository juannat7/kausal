import torch
import itertools
from .base import BaseObservables

class PolynomialFeatures(BaseObservables):
    """
    Returns features of polynomials

    Parameters:
        degree (int): the order of maximum power.

    Returns
        Identity (torch.Tensor): The Identity transforms.
    """

    def __init__(self, degree = 2):
        super().__init__()
        self.degree = degree

    def fit(self, x = None, y = None, **kwargs):
        return self
    
    def forward(self, X):
        X = self.validate(X)
        N, D = X.shape
        
        Psi = []
        
        # For each [1, degree + 1]
        for d in range(1, self.degree + 1):

            # Indices is a tuple e.g., (0, 2) meaning we multiply X[0] * X[2]
            I = itertools.combinations_with_replacement(range(N), d)
            
            for i in I:
                
                Psi.append(
                    torch.prod(X[list(i), :], dim=0, keepdim=True)
                )
        
        return torch.cat(Psi, dim=0)
