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
    return jnp.cos(jnp.dot(W.T, X) + b)


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
    Compute causal loss for both causal and non-causal directions.

    Parameters:
        cause: Causing data states.
        effect: Effected data states.

    Returns:
        causal_model_error: Causal direction error.
        non_causal_model_error: Non-causal direction error.
        omega_marginal_causal, omega_joint_causal: Forecasts in causal direction.
        omega_marginal_non_causal, omega_joint_non_causal: Forecasts in non-causal direction.
    """
    N, D = cause.shape
    cause_effect = jnp.concatenate([cause, effect], axis=0)

    # Generate the omega and its shifts
    omega_C, omega_C_t = cause[:, :-t], cause[:, t:]
    omega_E, omega_E_t = effect[:, :-t], effect[:, t:]
    omega_CE, omega_CE_t = cause_effect[:, :-t], cause_effect[:, t:]

    # Compute their DMD transforms
    psi_C, psi_C_t = random_fourier_features(omega_C), random_fourier_features(omega_C_t)
    psi_E, psi_E_t = random_fourier_features(omega_E), random_fourier_features(omega_E_t)
    psi_CE, psi_CE_t = random_fourier_features(omega_CE), random_fourier_features(omega_CE_t)

    # Step 1: Causal Direction (C -> E)

    ## Compute K
    K_marginal_causal = dynamic_mode_decomposition(
        jnp.concatenate([omega_E, psi_E], axis=0), omega_E_t
    )
    K_joint_causal = dynamic_mode_decomposition(
        jnp.concatenate([omega_E, psi_CE], axis=0), omega_CE_t
    )

    ## Compute marginal / joint
    omega_marginal_causal = K_marginal_causal @ jnp.concatenate([omega_E, psi_E], axis=0)
    omega_joint_causal = K_joint_causal @ jnp.concatenate([omega_E, psi_CE], axis=0)

    ## Compute errors
    causal_err_marginal = jnp.square(omega_marginal_causal - omega_E_t).mean()
    causal_err_joint = jnp.square(omega_joint_causal[:N] - omega_E_t).mean()
    causal_model_error = causal_err_joint - causal_err_marginal

    # Step 2: Non-Causal Direction (E -> C)

    ## Compute K
    K_marginal_non_causal = dynamic_mode_decomposition(
        jnp.concatenate([omega_C, psi_C], axis=0), omega_C_t
    )
    K_joint_non_causal = dynamic_mode_decomposition(
        jnp.concatenate([omega_C, psi_CE], axis=0), omega_CE_t
    )

    ## Compute marginal / joint
    omega_marginal_non_causal = K_marginal_non_causal @ jnp.concatenate([omega_C, psi_C], axis=0)
    omega_joint_non_causal = K_joint_non_causal @ jnp.concatenate([omega_C, psi_CE], axis=0)

    ## Compute errors
    non_causal_err_marginal = jnp.square(omega_marginal_non_causal - omega_C_t).mean()
    non_causal_err_joint = jnp.square(omega_joint_non_causal[:N] - omega_C_t).mean()
    non_causal_model_error = non_causal_err_joint - non_causal_err_marginal

    return (
        causal_model_error,
        non_causal_model_error,
        omega_marginal_causal,
        omega_joint_causal,
        omega_marginal_non_causal,
        omega_joint_non_causal,
    )