import jax
import jax.numpy as jnp

def random_fourier_features(X, M=500, key=jax.random.PRNGKey(42)):
    """
    Generate Random Fourier Features for input data X 
        Reference: Equation 10

    Parameters:
        X (jax.numpy.ndarray): Input data of shape (N, D).
        M (int): Number of random Fourier features.
        key (jax.random.PRNGKey): Random seed.

    Returns:
        jax.numpy.ndarray: Transformed data of shape (n_samples, M).
    """
    N, D = X.shape
    key_W, key_b = jax.random.split(key)

    W = jax.random.normal(key_W, shape=(N, M)) # Random projection matrix
    b = jax.random.uniform(key_b, shape=(M, D)) # Random biases

    # Compute RFF
    Z = jnp.cos(jnp.dot(W.T, X) + b)
    return Z


def dynamic_mode_decomposition(Psi, Psi_t):
    """
    Perform Dynamic Mode Decomposition (DMD) in the original space.

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


def compute_causal_loss(cause, effect, t=1):
    """
    Compute causal loss given the cause, effect data, and the time shift
    
    Parameters:
        cause: causing data states.
        effect: effected data states.

    Returns:
        model_error: computed based on the difference between marginal and joint errors.
        omega_marginal: omega when considering only the effect (no cause)
        omega_joint: omega when considering both the cause and effect
    """
    # Generate the omega and its shifts
    omega_C, omega_C_t = cause[:, :-t], cause[:, t:]
    omega_E, omega_E_t = effect[:, :-t], effect[:, t:]
    
    # Compute their DMD transforms 
    psi_C, psi_C_t = random_fourier_features(omega_C), random_fourier_features(omega_C_t)
    psi_E, psi_E_t = random_fourier_features(omega_E), random_fourier_features(omega_E_t)
    
    # Derive Koopman operators, K_t
    K_marginal = dynamic_mode_decomposition(
        jnp.concatenate([omega_E, psi_E], axis=0), omega_E_t
    )
    
    K_joint = dynamic_mode_decomposition(
        jnp.concatenate([omega_E, psi_E, psi_C], axis=0), omega_E_t
    )

    # Project to the data space
    omega_marginal = K_marginal @ jnp.concatenate([omega_E, psi_E], axis=0)
    omega_joint = K_joint @ jnp.concatenate([omega_E, psi_E, psi_C], axis=0)

    # Compute causal loss
    marginal_err = jnp.square(omega_marginal - omega_E_t).mean() 
    joint_err = jnp.square(omega_joint - omega_E_t).mean() 
    model_error = marginal_err - joint_err
    
    return model_error, omega_marginal, omega_joint
