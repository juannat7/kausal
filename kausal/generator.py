import numpy as np
import torch
import torch.nn.functional as F

def coupled_rossler(t, state, args):
    """
    Coupled Rossler dynamics solved 
        Reference: Equation 1

    Parameters:
        t: time.
        state: system states (6 variables / degree of freedom).
        args: scalar parameters, including c1 c2 as the coupling terms.

    Returns:
        tendency: system state tendencies.
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

    return torch.tensor([dx1, dy1, dz1, dx2, dy2, dz2])
    

def reaction_diffusion_2d(t, state, args):
    """
    Coupled reaction-diffusion in 2D field

    Parameters:
        t: time.
        state: system states.
        args: scalar parameters, including beta and gamma coupling terms.

    Returns:
        tendency: system state tendencies.
    """
    
    def _laplacian_2d(T, dx, dy):
        """Helper function to compute 2D Laplacian"""
        T_padded = F.pad(T.unsqueeze(0), pad=(1, 1, 1, 1), mode = "replicate").squeeze()  # Neumann BC
        laplacian_x = (T_padded[2:, 1:-1] - 2 * T_padded[1:-1, 1:-1] + T_padded[:-2, 1:-1]) / dx**2
        laplacian_y = (T_padded[1:-1, 2:] - 2 * T_padded[1:-1, 1:-1] + T_padded[1:-1, :-2]) / dy**2
        return laplacian_x + laplacian_y
    
    D_u, D_v, a, b, beta, gamma, Nx, Ny, dx, dy = args
    u = state[:Nx * Ny].reshape((Nx, Ny))
    v = state[Nx * Ny:].reshape((Nx, Ny))

    # Diffusion terms
    laplacian_u = _laplacian_2d(u, dx, dy)
    laplacian_v = _laplacian_2d(v, dx, dy)

    # Reaction terms with coupling
    f_u = -u * (u - a) * (u - 1) + beta * v
    f_v = -v * (v - b) * (v - 1) + gamma * u

    # Time derivatives
    du_dt = D_u * laplacian_u + f_u
    dv_dt = D_v * laplacian_v + f_v

    return torch.cat([du_dt.ravel(), dv_dt.ravel()])


def enso(t, state, args):
    """
    El-Nino Southern Oscillation (ENSO), recharge oscillators.
    Reference: 
        [1] https://www.aoml.noaa.gov/phod/docs/2004_Wang_Picaut.pdf
        [2] https://journals.ametsoc.org/view/journals/atsc/54/7/1520-0469_1997_054_0811_aeorpf_2.0.co_2.xml

    Parameters:
        t: time.
        state: system states.
        args: scalar parameters.

    Returns:
        tendency: system state tendencies.
    """
    T, h = state
    r, alpha, b0, c, gamma, mu, eps = args

    # Define the equations
    dT = -r * T - mu * alpha * b0 * h - eps * T ** 3
    dh = gamma * T + (gamma * mu * b0 - c) * h

    return torch.tensor([dT, dh])
