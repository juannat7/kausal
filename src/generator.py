import jax.numpy as jnp

def coupled_rossler(t, state, args):
    """
    Coupled Rossler dynamics solved 
        Reference: Equation 1

    Parameters:
        t: time
        state: system state (6 variables / degree of freedom)
        args: scalar parameters, including c1 c2 as the coupling terms

    Returns:
        tendency: system state tendencies
    """
    x1, y1, z1, x2, y2, z2 = state
    phi1, phi2, a, b, d, c1, c2 = args

    # Define the equations
    dx1 = -phi1 * y1 - z1
    dy1 = phi1 * x1 + a * y1 + c1 * (y2 - y1)
    dz1 = b + z1 * (x1 - d)

    dx2 = -phi2 * y2 - z2
    dy2 = phi2 * x2 + a * y2 + c2 * (y1 - y2)
    dz2 = b + z2 * (x2 - d)

    return jnp.array([dx1, dy1, dz1, dx2, dy2, dz2])
    