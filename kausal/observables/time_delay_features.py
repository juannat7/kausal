import torch
import itertools
from .base import BaseObservables

class TimeDelayFeatures(BaseObservables):
    """
    Returns time delay features such as [x(t), x(t-\\Delta$ t), x(t-2\\Delta t), ...]

    Parameters:
        n_delays (int): The size of time delay embedding.

    Returns
        self.
    """

    def __init__(self, n_delays = 2):
        super().__init__()
        self.n_delays = n_delays

    def fit(self, x = None, y = None, **kwargs):
        return self
    
    def forward(self, X):
        X = self.validate(X)
        N, D = X.shape
        
        Psi = torch.empty((
            int(N * (1 + self.n_delays)),
            int(D)
        ))

        Psi[:N, :] = X
        
        # Extract time delay embedding for each timestep index 
        for i in range(D):
            Psi[N:, i] = X[:, self._delay_inds(i)].flatten()

        return Psi

    def _delay_inds(self, index):
        """
        Private method to get the indices for the delayed data.
        """
        return (index - torch.arange(1, self.n_delays + 1)).clamp(min=0)
        
