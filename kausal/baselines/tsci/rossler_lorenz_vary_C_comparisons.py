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
from statsmodels.tsa.stattools import grangercausalitytests

from cdt.causality.pairwise import IGCI
from causallearn.search.FCMBased.ANM.ANM import ANM
from patched_reci import RECI
import warnings

warnings.filterwarnings(
    "ignore", message="verbose is deprecated since functions should not print results"
)

import faulthandler

faulthandler.enable()

Cs = np.linspace(0.0, 3.0, 5)
N_trials = 10
res_x2ys = []
res_x2ys_cutoffs = []
res_y2xs = []
res_y2xs_cutoffs = []
res_x2y_ccms = []
res_y2x_ccms = []
res_x2y_mis = []
res_y2x_mis = []

seed_time = int(time.time())
np.random.seed(seed_time)
verbose = True

for trial in range(N_trials):
    res_x2y = []
    res_y2x = []
    res_x2y_cutoff = []
    res_y2x_cutoff = []
    res_x2y_ccm = []
    res_y2x_ccm = []
    res_x2y_mi = []
    res_y2x_mi = []

    for C in tqdm(Cs):
        # Generate Data
        z0 = np.array([-0.82, -0.8, -0.24, 10.01, -12.19, 10.70])
        z0 = z0 + np.random.randn(*z0.shape) * 1e-3
        x, y = generate_lorenz_rossler(np.linspace(0, 110, 8000), z0, C)

        # print(np.sqrt(np.mean(x[:, 1] ** 2)))
        x_signal = y[:, 0].reshape(-1, 1)
        y_signal = x[:, 1].reshape(-1, 1)

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

        dx_dt = discrete_velocity(x_signal)
        dy_dt = discrete_velocity(y_signal)

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

        if verbose:
            print("TSCI (CS) X -> Y", np.mean(r_x2y))
            print("TSCI (CS) Y -> X", np.mean(r_y2x))

        res_x2y.append(np.mean(r_x2y))
        res_x2y_cutoff.append(
            2 / np.sqrt(Q_x) * np.sqrt(autocorr_new(r_x2y) / len(r_x2y))
        )
        res_y2x.append(np.mean(r_y2x))
        res_y2x_cutoff.append(
            2 / np.sqrt(Q_y) * np.sqrt(autocorr_new(r_y2x) / len(r_y2x))
        )

        r_x2y, r_y2x = tsci.tsci_nn(
            x_state,
            y_state,
            dx_state,
            dy_state,
            fraction_train=0.8,
            use_mutual_info=True,
        )

        res_x2y_mi.append(r_x2y)
        res_y2x_mi.append(r_y2x)

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
        lib_lens = [100, len_tr]
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
        lib_lens = [100, len_tr]
        CCM.fit(x1tr, x2tr)
        x1p, x2p = CCM.predict(x1te, x2te, lib_lengths=lib_lens)
        sc1, sc2 = CCM.score()
        res_y2x_ccm.append(sc1[1])  # - sc1[0])

        gc_y2x = grangercausalitytests(
            np.stack([x_signal.squeeze(), y_signal.squeeze()], axis=-1),
            3,
            verbose=False,
        )

        gc_x2y = grangercausalitytests(
            np.stack([y_signal.squeeze(), x_signal.squeeze()], axis=-1),
            3,
            verbose=False,
        )

        reci = RECI()
        reci_result = reci.predict_proba(
            (x_signal.reshape((-1, 1)), y_signal.reshape((-1, 1)))
        )

        igci = IGCI()
        igci_result = igci.predict_proba(
            (x_signal.reshape((-1, 1)), y_signal.reshape((-1, 1)))
        )

        anm = ANM()
        anm_x2y, anm_y2x = anm.cause_or_effect(
            x_signal[::10].reshape((-1, 1)), y_signal[::10].reshape((-1, 1))
        )

        if verbose:
            print("TSCI (MI) X -> Y", r_x2y)
            print("TSCI (MI) Y -> X", r_y2x)
            print("CCM X -> Y", res_x2y_ccm[-1])
            print("CCM Y -> X", res_y2x_ccm[-1])
            print(
                "GC p-value Y -> X",
                gc_y2x[3][0]["ssr_ftest"][1],
            )
            print("GC p-value X -> Y", gc_x2y[3][0]["ssr_ftest"][1])
            print("RECI Result", reci_result)
            print("IGCI Result", igci_result)
            print("ANM Result X -> Y", anm_x2y)
            print("ANM Result X -> Y", anm_y2x)

    res_x2ys.append(res_x2y)
    res_x2ys_cutoffs.append(res_x2y_cutoff)
    res_y2xs.append(res_y2x)
    res_y2xs_cutoffs.append(res_y2x_cutoff)
    res_x2y_ccms.append(res_x2y_ccm)
    res_y2x_ccms.append(res_y2x_ccm)
    res_x2y_mis.append(res_x2y_mi)
    res_y2x_mis.append(res_y2x_mi)


res_x2ys_np = np.array(res_x2ys)


res_y2xs_np = np.array(res_y2xs)
res_x2ys_cutoffs_np = np.array(res_x2ys_cutoffs)
res_y2xs_cutoffs_np = np.array(res_y2xs_cutoffs)
res_x2y_ccms_np = np.array(res_x2y_ccms)
res_y2x_ccms_np = np.array(res_y2x_ccms)
res_x2y_mis_np = np.array(res_x2y_mis)
res_y2x_mis_np = np.array(res_y2x_mis)


np.savez(
    f"results/{seed_time}_vary_C.npz",
    res_x2ys_np,
    res_y2xs_np,
    res_x2ys_cutoffs_np,
    res_y2xs_cutoffs_np,
    res_x2y_ccms_np,
    res_y2x_ccms_np,
    res_x2y_mis_np,
    res_y2x_mis_np,
)
