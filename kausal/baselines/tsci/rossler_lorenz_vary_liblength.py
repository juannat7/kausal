import tsci
from tqdm import tqdm
import numpy as np
from utils import (
    generate_lorenz_rossler,
    lag_select,
    false_nearest_neighbors,
    delay_embed,
    discrete_velocity,
    autocorr_new,
)
import skccm as ccm  # type: ignore
import time

C = 1.0
sigma = 0.0
N_trials = 20
res_x2ys = []
res_x2ys_cutoffs = []
res_y2xs = []
res_y2xs_cutoffs = []
res_x2y_ccms = []
res_y2x_ccms = []
liblens = np.linspace(100, 6_000, 10)

seed_time = int(time.time())
np.random.seed(seed_time)


def estimate_threshold(d, p=0.95):
    x = np.random.randn(10_000, d)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = np.random.randn(10_000, d)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)

    return np.quantile(np.sum(x * y, axis=1), 0.95)


for trial in range(N_trials):
    res_x2y = []
    res_y2x = []
    res_x2y_cutoff = []
    res_y2x_cutoff = []
    res_x2y_ccm = []
    res_y2x_ccm = []

    for liblen in tqdm(liblens):
        # Generate Data
        z0 = np.array([-0.82, -0.8, -0.24, 10.01, -12.19, 10.70])
        z0 = z0 + np.random.randn(*z0.shape) * 1e-3
        x, y = generate_lorenz_rossler(np.linspace(0, 110, 8000), z0, C)

        x_signal = y[:, 0].reshape(-1, 1)
        y_signal = x[:, 1].reshape(-1, 1)
        x_signal = (x_signal - np.mean(x_signal, axis=0, keepdims=True)) / np.std(
            x_signal, axis=0, keepdims=True
        )
        y_signal = (y_signal - np.mean(y_signal, axis=0, keepdims=True)) / np.std(
            y_signal, axis=0, keepdims=True
        )
        x_signal = x_signal + np.random.randn(*x_signal.shape) * sigma
        y_signal = y_signal + np.random.randn(*y_signal.shape) * sigma

        # Get CCM hyperparameters and create delay embeddings
        tau_x = lag_select(x_signal, theta=0.5)
        tau_y = lag_select(y_signal, theta=0.5)
        Q_x_ccm = false_nearest_neighbors(x_signal, tau_x, fnn_tol=0.005)
        Q_y_ccm = false_nearest_neighbors(y_signal, tau_y, fnn_tol=0.005)
        Q_x = false_nearest_neighbors(x_signal, tau_x, fnn_tol=0.005)
        Q_y = false_nearest_neighbors(y_signal, tau_y, fnn_tol=0.005)

        x_state = delay_embed(x_signal, tau_x, Q_x)
        y_state = delay_embed(y_signal, tau_y, Q_y)
        truncated_length = min(x_state.shape[0], y_state.shape[0]) - 100
        x_state = x_state[-truncated_length:]
        y_state = y_state[-truncated_length:]

        dx_dt = discrete_velocity(x_signal, smooth=True)
        dy_dt = discrete_velocity(y_signal, smooth=True)

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
            lib_length=int(liblen),
        )

        res_x2y.append(np.mean(r_x2y))
        res_x2y_cutoff.append(
            estimate_threshold(Q_x) * np.sqrt(autocorr_new(r_x2y) / len(r_x2y))
        )
        res_y2x.append(np.mean(r_y2x))
        res_y2x_cutoff.append(
            estimate_threshold(Q_y) * np.sqrt(autocorr_new(r_y2x) / len(r_y2x))
        )

        ###########################
        ####    Perform CCM    ####
        ###########################
        e1 = ccm.Embed(x_signal.squeeze())
        e2 = ccm.Embed(y_signal.squeeze())
        X1 = e1.embed_vectors_1d(tau_y, Q_y_ccm)
        X2 = e2.embed_vectors_1d(tau_y, Q_y_ccm)
        x1tr, x1te, x2tr, x2te = ccm.utilities.train_test_split(X1, X2, percent=0.8)
        CCM = ccm.CCM()
        len_tr = len(x1tr)
        lib_lens = [100, int(liblen)]
        CCM.fit(x1tr, x2tr)
        x1p, x2p = CCM.predict(x1te, x2te, lib_lengths=lib_lens)
        sc1, sc2 = CCM.score()
        res_x2y_ccm.append(sc1[1])  # - sc1[0])

        e1 = ccm.Embed(y_signal.squeeze())
        e2 = ccm.Embed(x_signal.squeeze())
        X1 = e1.embed_vectors_1d(tau_x, Q_x_ccm)
        X2 = e2.embed_vectors_1d(tau_x, Q_x_ccm)
        x1tr, x1te, x2tr, x2te = ccm.utilities.train_test_split(X1, X2, percent=0.8)
        CCM = ccm.CCM()
        len_tr = len(x1tr)
        lib_lens = [100, int(liblen)]
        CCM.fit(x1tr, x2tr)
        x1p, x2p = CCM.predict(x1te, x2te, lib_lengths=lib_lens)
        sc1, sc2 = CCM.score()
        res_y2x_ccm.append(sc1[1])  # - sc1[0])

    res_x2ys.append(res_x2y)
    res_x2ys_cutoffs.append(res_x2y_cutoff)
    res_y2xs.append(res_y2x)
    res_y2xs_cutoffs.append(res_y2x_cutoff)
    res_x2y_ccms.append(res_x2y_ccm)
    res_y2x_ccms.append(res_y2x_ccm)

res_x2ys_np = np.array(res_x2ys)
res_y2xs_np = np.array(res_y2xs)
res_x2ys_cutoffs_np = np.array(res_x2ys_cutoffs)
res_y2xs_cutoffs_np = np.array(res_y2xs_cutoffs)
res_x2y_ccms_np = np.array(res_x2y_ccms)
res_y2x_ccms_np = np.array(res_y2x_ccms)

np.savez(
    f"results/{seed_time}_vary_liblength.npz",
    res_x2ys_np,
    res_y2xs_np,
    res_x2ys_cutoffs_np,
    res_y2xs_cutoffs_np,
    res_x2y_ccms_np,
    res_y2x_ccms_np,
)
