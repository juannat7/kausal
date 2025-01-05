import numpy as np
import torch
import torch.nn.functional as F

from src.observables import RandomFourierFeatures
from src.utils import (
    validate
)

def dynamic_mode_decomposition(Psi, Psi_t):
    """TODO(juan): refactor to a common regression class
    Perform Dynamic Mode Decomposition (DMD).

    Parameters:
        Psi: Observables at current time.
        Psi_t: Observables at shifted time.

    Returns:
        K_t: Koopman operator in the original space.
    """
    Psi_inverse = torch.linalg.pinv(Psi)
    K_t = Psi_t @ Psi_inverse
    return K_t


def compute_transforms(cause, effect, t=1, **kwargs):
    """
    Compute Koopman transforms from states to observables

    Parameters:
        cause (torch.Tensor): The cause variables to be tested.
        effect (torch.Tensor): the effect variables to be tested.
        t (int): The time shift.

    Returns:
        observables (torch.Tensor): lifted effect (marginal) and effect-cause (joint).
        transform_fns (Callable): transformation functions used for the marginal and joint models.
    """
    
    effect_cause = torch.cat([effect, cause], axis=0)

    # Generate the omega and its shifts
    omega_E, omega_Et = effect[..., :-t], effect[..., t:]
    omega_EC, omega_ECt = effect_cause[..., :-t], effect_cause[..., t:]

    # Retrieve transformation functions
    marginal_transform_fn = kwargs.get("marginal_model") or RandomFourierFeatures()
    joint_transform_fn = kwargs.get("joint_model") or RandomFourierFeatures()

    # Compute their transforms
    psi_E, psi_Et = marginal_transform_fn(omega_E), marginal_transform_fn(omega_Et) # Marginal transforms
    psi_EC, psi_ECt = joint_transform_fn(omega_EC), joint_transform_fn(omega_ECt)   # Joint transforms
        
    return (
        validate(omega_E), validate(omega_Et), validate(omega_EC), validate(omega_ECt), 
        validate(psi_E), validate(psi_Et), validate(psi_EC), validate(psi_ECt),
        marginal_transform_fn, joint_transform_fn
    )


def compute_K(omega_E, psi_E, omega_EC, psi_EC, omega_Et):
    """
    Compute approximation to Koopman operator with regression algorithm (e.g., DMD)
    """
    K_marginal = dynamic_mode_decomposition(torch.cat([omega_E, psi_E], axis=0), omega_Et)
    K_joint = dynamic_mode_decomposition(torch.cat([omega_E, psi_EC], axis=0), omega_Et)

    return (
        K_marginal, K_joint
    )


def compute_causal_loss(cause, effect, t=1, **kwargs):
    """
    Compute causal loss through marginal and joint models.

    Parameters:
        cause: Causing data states.
        effect: Effected data states.
        t: Time shifts.

    Returns:
        causal_error: Causal error to indicate causal direction.
        omega_marginal: Marginal estimates.
        omega_joint: Joint estimates.
    """
    marginal_model = kwargs.get("marginal_model", None)
    joint_model = kwargs.get("joint_model", None)

    # Step 1: Compute observables
    omega_E, omega_Et, omega_EC, omega_ECt, psi_E, psi_Et, psi_EC, psi_ECt, _, _ = compute_transforms(
        cause, effect, t, marginal_model = marginal_model, joint_model = joint_model
    )
        
    # Step 2: Approximate Koopman operator
    K_marginal, K_joint = compute_K(omega_E, psi_E, omega_EC, psi_EC, omega_Et)

    # Step 3: Evaluate marginal / joint models
    omega_marginal = K_marginal @ torch.cat([omega_E, psi_E], axis=0)
    omega_joint = K_joint @ torch.cat([omega_E, psi_EC], axis=0)
    
    # Compute errors
    causal_err_marginal = F.mse_loss(omega_marginal, omega_Et)
    causal_err_joint = F.mse_loss(omega_joint, omega_Et)
    causal_error = causal_err_marginal - causal_err_joint

    return (
        causal_error,
        omega_marginal,
        omega_joint
    )


def conditional_forecasting(cause, effect, t=1, **kwargs):
    """
    Compute conditional forecasting by 1) computing transforms, 2) estimating K, 3) performing conditional forecasting.
    Parameters:
        cause: Causing data states.
        effect: Effected data states.
        t: Time shifts.

    Returns:
        omega_marginal: Marginal estimates.
        omega_joint: Joint estimates.
    """
    
    effect_cause = torch.cat([effect, cause], axis=0)
    D = effect.shape[-1]

    marginal_model = kwargs.get("marginal_model", None)
    joint_model = kwargs.get("joint_model", None)
    n_train = kwargs.get("n_train", D)

    # Step 1: Compute observables
    omega_E, omega_Et, omega_EC, omega_ECt, psi_E, psi_Et, psi_EC, psi_ECt, marginal_transform_fn, joint_transform_fn = compute_transforms(
        cause[..., :n_train], effect[..., :n_train], t, marginal_model = marginal_model, joint_model = joint_model
    )

    # Step 2: Approximate Koopman operator
    K_marginal, K_joint = compute_K(omega_E, psi_E, omega_EC, psi_EC, omega_Et)

    
    # Step 3: Conditional forecasting given marginal / joint models
    omega_marginal, omega_joint = [effect[..., 0:1]], [effect[..., 0:1]]

    for d in range(D - t):

        ## Marginal model
        omega_marginal.append(
            K_marginal @ torch.cat([omega_marginal[-1], marginal_transform_fn(effect[..., d : d+1])], axis=0)
        )
    
        ## Joint model
        omega_joint.append(
            K_joint @ torch.cat([omega_joint[-1], joint_transform_fn(effect_cause[..., d : d+1])], axis=0)
        )
        
            
    omega_marginal = torch.stack(omega_marginal).squeeze().T
    omega_joint = torch.stack(omega_joint).squeeze().T

    return (
        omega_marginal, 
        omega_joint
    )