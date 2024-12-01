import numpy as np
import jax
import jax.numpy as jnp
import pykoopman as pk

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


def compute_transforms(cause, effect, t=1):
    """Compute cause-effect transforms"""
    N, D = cause.shape
    
    effect_cause = jnp.concatenate([effect, cause], axis=0)

    # Generate the omega and its shifts
    omega_E, omega_Et = effect[:, :-t], effect[:, t:]
    omega_EC, omega_ECt = effect_cause[:, :-t], effect_cause[:, t:]

    # Compute their DMD transforms
    psi_E, psi_Et = random_fourier_features(omega_E), random_fourier_features(omega_Et)
    psi_EC, psi_ECt = random_fourier_features(omega_EC), random_fourier_features(omega_ECt)

    return omega_E, omega_Et, omega_EC, omega_ECt, psi_E, psi_Et, psi_EC, psi_ECt



def compute_K(omega_E, psi_E, omega_EC, psi_EC, omega_Et):
    """Compute approximation to Koopman operator with DMD algorithm"""

    # Compute marginal model
    K_marginal = dynamic_mode_decomposition(
        jnp.concatenate([omega_E, psi_E], axis=0), omega_Et
    )

    # Compute joint model
    K_joint = dynamic_mode_decomposition(
        jnp.concatenate([omega_E, psi_EC], axis=0), omega_Et
    )

    return K_marginal, K_joint


def compute_causal_loss(cause, effect, t=1):
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
    # Compute transforms
    omega_E, omega_Et, omega_EC, omega_ECt, psi_E, psi_Et, psi_EC, psi_ECt = compute_transforms(cause, effect, t)

    # Compute K
    K_marginal, K_joint = compute_K(omega_E, psi_E, omega_EC, psi_EC, omega_Et)
    
    # Compute marginal / joint
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
