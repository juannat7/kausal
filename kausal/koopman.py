import abc
import numpy as np
import torch
import torch.nn.functional as F

from .observables import RandomFourierFeatures
from .regressors import PINV, DMD
from .utils import validate

class Kausal(abc.ABC):
    """
    Causal inference with deep Koopman operator.

    Parameters:
        regressor (BaseRegressor): Regressor object, defaults to PINV.
        marginal_observable (BaseObservables): Observables object for the marginal model, defaults to RFF.
        joint_observable (BaseObservables): Observables object for the joint model, defaults to RFF.
        cause (torch.Tensor): Causal variables to be tested, shape (n_channels, *, n_timestep).
        effect (torch.Tensor): Effect variables to be tested, shape (n_channels, *, n_timestep).

    Returns:
        self.
    """
    def __init__(
        self,
        regressor = PINV(),
        marginal_observable = RandomFourierFeatures(),
        joint_observable = RandomFourierFeatures(),
        cause = None,
        effect = None
        
    ):
        """Initialize lifting class"""
        super(Kausal, self).__init__()
        
        if cause is None or effect is None:
            raise ValueError("Cause-effect variables must be provided.")
        
        self.regressor = regressor
        self.marginal_observable = marginal_observable
        self.joint_observable = joint_observable        
        self.cause = cause
        self.effect = effect
        self.effect_cause = torch.cat([effect, cause], axis=0)
        
        *self.C, self.T = effect.shape
        self.device = effect.device

    
    def fit(
        self, 
        n_train = None,
        **kwargs
    ):
        """
        Fit the observable functions, if needed, such as when using deep learning approximation.

        Parameters:
            n_train (NoneType, int): The number of first n samples used for training the observable functions.
            transform_func (NoneType, Callable): The transform function to preprocess training data.

        Returns:
            marginal_loss (torch.Tensor): The marginal observable loss over n_epochs.
            joint_loss (torch.Tensor): The joint observable loss over n_epochs.
        """
        # Prepare dataset
        cause = self.cause[..., :n_train] if n_train is not None else self.cause
        effect = self.effect[..., :n_train] if n_train is not None else self.effect
        effect_cause = torch.cat([effect, cause], axis=0)

        # Fit observables
        marginal_loss = self.marginal_observable.fit(x = effect, y = effect, **kwargs)
        joint_loss = self.joint_observable.fit(x = effect_cause, y = effect, **kwargs)
        return marginal_loss, joint_loss
    
    
    def evaluate(
        self, 
        time_shift = 1,
        **kwargs
    ):
        """
        Evaluate causal strength through marginal/joint difference formulation in the observable space.
    
        Parameters:
            time_shift (int: 1): Time shifts.
    
        Returns:
            causal_error: Causal error in the cause --> effect variables.
        """
        
        # Step 1: Observable transforms 
        wE, wEt, pE, pEC = self._transform_state(
            cause = self.cause, 
            effect = self.effect, 
            time_shift = time_shift
        )
            
        # Step 2: Approximate Koopman operator
        Km, Kj = self._estimate_koopman(wE, wEt, pE, pEC)
    
        # Step 3: Evaluate marginal / joint models
        wm = self._koopman_step(Km, torch.cat([wE, pE], axis=0))
        wj = self._koopman_step(Kj, torch.cat([wE, pEC], axis=0))
        
        # Step 3: Compute errors (marginal - joint)
        return F.mse_loss(wm, wEt) - F.mse_loss(wj, wEt)
            

    def forecast(
        self, 
        n_train = None,
        time_shift = 1,
        **kwargs
    ):
        """
        Compute conditional forecasting, by:  1) computing transforms, 2) estimating K,  3) performing conditional forecasting.
        
        Parameters:
            n_train (NoneType, int):  The number of first n samples used for training the observable functions.
            time_shift (int: 1): Time shifts.
    
        Returns:
            wm (torch.Tensor): Marginal estimates.
            wj (torch.Tensor): Joint estimates.
        """
    
        # Step 1: Compute observables
        wE, wEt, pE, pEC = self._transform_state(
            cause = self.cause[..., :n_train] if n_train is not None else self.cause, 
            effect = self.effect[..., :n_train] if n_train is not None else self.effect, 
            time_shift = time_shift
        )
    
        # Step 2: Approximate Koopman operator
        Km, Kj = self._estimate_koopman(wE, wEt, pE, pEC)
    
        
        # Step 3: Conditional forecasting given marginal / joint models
        wm = [validate(self.effect[..., 0:1])]
        wj = [validate(self.effect[..., 0:1])]
    
        for d in range(self.T - time_shift):
    
            ## Marginal model
            wm.append(
                self._koopman_step(
                    K = Km,
                    w = torch.cat([wm[-1], self.marginal_observable(self.effect[..., d : d+1])], axis=0)
                )
            )
        
            ## Joint model
            wj.append(
                self._koopman_step(
                    K = Kj,
                    w = torch.cat([wj[-1], self.joint_observable(self.effect_cause[..., d : d+1])], axis=0)
                )
            )
                
        wm, wj = torch.stack(wm).squeeze().T, torch.stack(wj).squeeze().T
        return wm, wj

        
    def _transform_state(
        self, 
        cause = None,
        effect = None,
        time_shift = 1
    ):
        """
        Lift high-dimensional, nonlinear states to the observable space.
    
        Parameters:
            cause (torch.Tensor): The cause variables.
            effect (torch.Tensor): The effect variables.
            time_shift (int: 1): Time shift.
    
        Returns:
            observables (torch.Tensor): lifted effect (marginal) and effect-cause (joint).
        """

        if cause is None or effect is None:
            raise ValueError("Cause-effect variables must be provided.")

        effect_cause = torch.cat([effect, cause], axis=0)

        # Generate shifted states
        wE, wEt = self._time_shift(x = effect, time_shift = time_shift)
        wEC, wECt = self._time_shift(x = effect_cause, time_shift = time_shift)
    
        # Compute their transforms, psi
        pE, pEt = self.marginal_observable(wE), self.marginal_observable(wEt) # Marginal transforms
        pEC, pECt = self.joint_observable(wEC), self.joint_observable(wECt)   # Joint transforms
            
        return (
            validate(wE), validate(wEt), validate(pE), validate(pEC)
        )
    
    
    def _estimate_koopman(self, wE, wEt, pE, pEC):
        """Compute approximation to Koopman operators with regression algorithm (e.g., DMD)."""
        Km = self.regressor(torch.cat([wE, pE], axis=0), wEt)
        Kj = self.regressor(torch.cat([wE, pEC], axis=0), wEt)
        return  Km, Kj
    
    
    def _time_shift(
        self,
        x,
        time_shift = 1
    ):
        """Shift x given `time_shift` parameter."""
        return x[..., :-time_shift], x[..., time_shift:]

    
    def _koopman_step(
        self,
        K,
        w
    ):
        """Evaluate the application of learned K on w, i.e., wt = K[w]."""
        return K @ w


        