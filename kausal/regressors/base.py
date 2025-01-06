import abc
import torch
import torch.nn as nn

from ..utils import validate

class BaseRegressor(nn.Module):
    """
    Abstract class for regressor.
    It inherits torch.nn Module
    """

    def __init__(self):
        """Initialize regressor"""
        super(BaseRegressor, self).__init__()
        

    @abc.abstractmethod
    def forward(self):
        """Regression method to estimate the Koopman operator given observables"""
        raise NotImplementedError
        