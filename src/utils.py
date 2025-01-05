import math

def validate(x): 
    """
    Flatten ND state/observables to valid 2D matrix with ND/T columns/rows.
    This is used to e.g., compute Koopman operator.
    """
    *ND, T = x.shape # T is the number of timesteps
    return x.reshape(math.prod(ND), T)