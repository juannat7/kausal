import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from src.nn import *
from src.utils import *

def random_fourier_features(X, M=500, key=jax.random.PRNGKey(42)):
    """
    Generate Random Fourier Features for input data X 
        Reference: Equation 10

    Parameters:
        X (jax.numpy.ndarray): Input data of shape (N, D).
        M (int): Number of random Fourier features.
        key (jax.random.PRNGKey): Random seed.

    Returns:
        jax.numpy.ndarray: Transformed data of shape (M, D).
    """
    N, D = X.shape
    key_W, key_b = jax.random.split(key)

    W = jax.random.normal(key_W, shape=(N, M))  # Random projection matrix
    b = jax.random.uniform(key_b, shape=(M, D)) # Random biases

    # Compute RFF
    return jnp.sqrt(2 / M) * jnp.cos(jnp.dot(W.T, X) + b)



def dynamic_mode_decomposition(Psi, Psi_t):
    """
    Perform Dynamic Mode Decomposition (DMD).

    Parameters:
        Psi: Observables at current time.
        Psi_t: Observables at shifted time.

    Returns:
        K_t: Koopman operator in the original space.
    """
    # Compute the pseudoinverse of Psi
    Psi_inverse = jnp.linalg.pinv(Psi)

    # Compute the Koopman operator K_t
    K_t = Psi_t @ Psi_inverse

    return K_t


def compute_transforms(cause, effect, t=1, **kwargs):
    """Compute cause-effect transforms"""
    effect_cause = jnp.concatenate([effect, cause], axis=0)

    # Generate the omega and its shifts
    omega_E, omega_Et = effect[..., :-t], effect[..., t:]
    omega_EC, omega_ECt = effect_cause[..., :-t], effect_cause[..., t:]

    # Retrieve transformation functions
    marginal_transform_fn = kwargs.get("marginal_model", random_fourier_features)
    joint_transform_fn = kwargs.get("joint_model", random_fourier_features)

    # Compute their transforms
    if isinstance(marginal_transform_fn, CNNEncoder):
        
        def _process_time_slices(model, data):
            return jax.vmap(model, in_axes=-1, out_axes=-1)(data)

        psi_E = _process_time_slices(marginal_transform_fn, omega_E)
        psi_Et = _process_time_slices(marginal_transform_fn, omega_Et)
        psi_EC = _process_time_slices(joint_transform_fn, omega_EC)
        psi_ECt = _process_time_slices(joint_transform_fn, omega_ECt)
    
    else:
        psi_E = marginal_transform_fn(omega_E)
        psi_Et = marginal_transform_fn(omega_Et)
        psi_EC = joint_transform_fn(omega_EC)
        psi_ECt = joint_transform_fn(omega_ECt)

    return omega_E, omega_Et, omega_EC, omega_ECt, psi_E, psi_Et, psi_EC, psi_ECt


def compute_K(omega_E, psi_E, omega_EC, psi_EC, omega_Et):
    """Compute approximation to Koopman operator with DMD algorithm"""
        
    K_marginal = dynamic_mode_decomposition(
        jnp.concatenate([omega_E, psi_E], axis=0), omega_Et
    )

    K_joint = dynamic_mode_decomposition(
        jnp.concatenate([omega_E, psi_EC], axis=0), omega_Et
    )

    return (
        K_marginal, 
        K_joint
    )


def compute_causal_loss(cause, effect, t=1, **kwargs):
    """
    Compute causal loss for both causal and non-causal directions.

    Parameters:
        cause: Causing data states.
        effect: Effected data states.
        t: Time shifts.

    Returns:
        causal_model_error: Causal direction error.
        omega_marginal: Marginal forecasts.
        omega_joint: Joint forecasts.
    """
    marginal_model = kwargs.get("marginal_model", None)
    joint_model = kwargs.get("joint_model", None)

    # Transform
    if (marginal_model == None) or (joint_model == None):

        omega_E, omega_Et, omega_EC, omega_ECt, psi_E, psi_Et, psi_EC, psi_ECt = compute_transforms(
            cause, effect, t
        )

    else:
        
        omega_E, omega_Et, omega_EC, omega_ECt, psi_E, psi_Et, psi_EC, psi_ECt = compute_transforms(
            cause, effect, t, marginal_model=marginal_model.encoder, joint_model=joint_model.encoder
        )

        if isinstance(marginal_model.encoder, CNNEncoder):
            omega_E, omega_Et, omega_EC, omega_ECt = flatten(omega_E), flatten(omega_Et), flatten(omega_EC), flatten(omega_ECt)
            psi_E, psi_Et, psi_EC, psi_ECt = flatten(psi_E), flatten(psi_Et), flatten(psi_EC), flatten(psi_ECt)
        
    # Compute DMD-based K
    K_marginal, K_joint = compute_K(omega_E, psi_E, omega_EC, psi_EC, omega_Et)

    # Evaluate omega
    omega_marginal = K_marginal @ jnp.concatenate([omega_E, psi_E], axis=0)
    omega_joint = K_joint @ jnp.concatenate([omega_E, psi_EC], axis=0)
    
    # Compute errors
    causal_err_marginal = jnp.mean(jnp.square(omega_marginal - omega_Et))
    causal_err_joint = jnp.mean(jnp.square(omega_joint - omega_Et))
    causal_model_error = causal_err_marginal - causal_err_joint

    return (
        causal_model_error,
        omega_marginal,
        omega_joint
    )


def conditional_forecasting(cause, effect, t=1, **kwargs):
    """
    Compute conditional forecasting by fitting K on a given training set, and iterating on a testing set
    given previous data point as the conditioning input.
    
    TODO: refactor for readability.
    """
    
    effect_cause = jnp.concatenate([effect, cause], axis=0)
    D = effect.shape[-1]
    is_cnn = False

    marginal_model = kwargs.get("marginal_model", None)
    joint_model = kwargs.get("joint_model", None)

    # Compute transforms
    if (marginal_model == None) or (joint_model == None):
        
        omega_E, omega_Et, omega_EC, omega_ECt, psi_E, psi_Et, psi_EC, psi_ECt = compute_transforms(
            cause, effect, t
        )

        marginal_transform_fn = random_fourier_features
        joint_transform_fn = random_fourier_features


    else:
        
        omega_E, omega_Et, omega_EC, omega_ECt, psi_E, psi_Et, psi_EC, psi_ECt = compute_transforms(
            cause, effect, t, marginal_model=marginal_model.encoder, joint_model=joint_model.encoder
        )

        if isinstance(marginal_model.encoder, CNNEncoder):

            is_cnn = True
            
            omega_E, omega_Et, omega_EC, omega_ECt = flatten(omega_E), flatten(omega_Et), flatten(omega_EC), flatten(omega_ECt)
            psi_E, psi_Et, psi_EC, psi_ECt = flatten(psi_E), flatten(psi_Et), flatten(psi_EC), flatten(psi_ECt)

        marginal_transform_fn = marginal_model.encoder
        joint_transform_fn = joint_model.encoder


    # Approximate K by psuedo-inverse method
    K_marginal, K_joint = compute_K(omega_E, psi_E, omega_EC, psi_EC, omega_Et)

    
    if is_cnn:

        # Compute marginal / joint on all data including test set
        omega_marginal, omega_joint = [flatten(effect[..., 0:1])], [flatten(effect[..., 0:1])]

        
        for d in range(D - t):
            
             # Marginal
            omega_marginal.append(
                K_marginal @ jnp.concatenate([flatten(omega_marginal[-1]), flatten(marginal_transform_fn(effect[..., d:d+1]))], axis=0)
            )
        
            # Joint
            omega_joint.append(
                K_joint @ jnp.concatenate([flatten(omega_joint[-1]), flatten(joint_transform_fn(effect_cause[..., d:d+1]))], axis=0)
            )

    else:
        
        # Compute marginal / joint on all data including test set
        omega_marginal, omega_joint = [effect[..., 0:1]], [effect[..., 0:1]]

        
        for d in range(D - t):

            # Marginal
            omega_marginal.append(
                K_marginal @ jnp.concatenate([omega_marginal[-1], marginal_transform_fn(effect[..., d:d+1])], axis=0)
            )
        
            # Joint
            omega_joint.append(
                K_joint @ jnp.concatenate([omega_joint[-1], joint_transform_fn(effect_cause[..., d:d+1])], axis=0)
            )
        
            
    
    omega_marginal = jnp.array(omega_marginal).squeeze().T
    omega_joint = jnp.array(omega_joint).squeeze().T

    return (
        omega_marginal, 
        omega_joint
    )