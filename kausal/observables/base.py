import abc
import torch
import torch.nn as nn

from ..utils import validate

class BaseObservables(nn.Module):
    """
    Abstract class for observables.
    It inherits torch.nn Module
    """

    def __init__(self):
        """Initialize lifting class"""
        super(BaseObservables, self).__init__()
        
    
    def validate(self, X):
        return validate(X)

    
    @abc.abstractmethod
    def fit(
        self, 
        x=None, 
        y=None, 
        **kwargs
    ):
        """Method to fit the observable functions"""
        raise NotImplementedError
        
    
    @abc.abstractmethod
    def forward(self, X):
        """Method to lift state X to observables f(X)"""
        raise NotImplementedError
        