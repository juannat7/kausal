import tsci
import numpy as np
from utils import (
    generate_lorenz_rossler,
    lag_select,
    false_nearest_neighbors,
    delay_embed,
    discrete_velocity,
)

# Generate Data from the Rossler-Lorenz system
C = 1.0
np.random.seed(0)
z0 = np.array([-0.82, -0.8, -0.24, 10.01, -12.19, 10.70])
z0 = z0 + np.random.randn(*z0.shape) * 1e-3
x, y = generate_lorenz_rossler(np.linspace(0, 110, 8000), z0, C)

x_signal = x[:, 1].reshape(-1, 1)
y_signal = y[:, 0].reshape(-1, 1)

# Get embedding hyperparameters and create delay embeddings
tau_x = lag_select(x_signal, theta=0.5)  # X lag
tau_y = lag_select(y_signal, theta=0.5)  # Y lag
Q_x = false_nearest_neighbors(x_signal, tau_x, fnn_tol=0.005)  # X embedding dim
Q_y = false_nearest_neighbors(y_signal, tau_y, fnn_tol=0.005)  # Y embedding dim

x_state = delay_embed(x_signal, tau_x, Q_x)
y_state = delay_embed(y_signal, tau_y, Q_y)
truncated_length = (
    min(x_state.shape[0], y_state.shape[0]) - 100
)  # Omit earliest samples
x_state = x_state[-truncated_length:]
y_state = y_state[-truncated_length:]

# Get velocities with (centered) finite differences
dx_dt = discrete_velocity(x_signal)
dy_dt = discrete_velocity(y_signal)

# Delay embed velocity vectors
dx_state = delay_embed(dx_dt, tau_x, Q_x)
dy_state = delay_embed(dy_dt, tau_y, Q_y)
dx_state = dx_state[-truncated_length:]
dx_state = dx_state
dy_state = dy_state[-truncated_length:]
dy_state = dy_state

############################
####    Perform TSCI    ####
############################
r_x2y, r_y2x = tsci.tsci_nn(
    x_state,
    y_state,
    dx_state,
    dy_state,
    fraction_train=0.8,
    use_mutual_info=False,
)

print(f"r_\u007bX -> Y\u007d: {np.mean(r_x2y):.2f}")
print(f"r_\u007bY -> X\u007d: {np.mean(r_y2x):.2f}")
