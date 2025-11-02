import torch
from .base import BaseObservables

class RandomFourierFeatures(BaseObservables):
    """
    Generate Random Fourier Features (RFF) for input data X 
        Reference: https://arxiv.org/pdf/2410.103; Equation 10

    Parameters:
        M (int): The number of random samples in the MC approximation.
        gamma (float): The scale of the Gaussian kernel.
        random_seed (int): Seeding for reproducibility.

    Returns
        RFF (torch.Tensor): The RFF transforms.
    """

    def __init__(
        self,
        M = 500,
        gamma = 1.0,
        random_seed = 42
    ):
        super().__init__()
        self.M = M
        self.gamma = gamma
        self.random_seed = random_seed

    def fit(self, x = None, y = None, **kwargs):
        return self
        
    def forward(
        self, 
        X
    ):
        torch.manual_seed(self.random_seed) 
        X = self.validate(X)
        N, D = X.shape

        # init projection matrix (W) and phase shifts (b)
        W = torch.randn(N, self.M)
        b = torch.rand(self.M, D)

        # get lifted feature vector
        return torch.sqrt(torch.tensor(2.0) * self.gamma / self.M) * torch.cos(torch.matmul(W.T, X) + b)
