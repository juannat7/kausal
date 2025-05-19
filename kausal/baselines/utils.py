"""
Baseline evaluation functions for causal tests and magnitudes.
"""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from lingam import DirectLiNGAM

from .tsci import tsci
from .tsci.utils import (
    false_nearest_neighbors,
    delay_embed,
    discrete_velocity,
)
from .neuralgc.neuralgc import train_model_ista, cLSTM
from .gvar.gvar import training_procedure_trgc

def normalize(x):
    """Min-max scaling for causal magnitude estimation"""
    return (x - x.min()) / (x.max() - x.min())


def center_slice(data, center, window_length):
    """Centered rolling window"""
    n = len(data)
    start = max(0, min(center - window_length // 2, n - window_length))
    return data[start : start + window_length]


def _stack(cause, effect, as_df=False):
    """Combine cause, effect dataset"""
    arr = torch.cat([effect, cause]).permute(1, 0).cpu().numpy()
    return pd.DataFrame(arr) if as_df else arr


def conditional_varlingam(cause, effect):
    """Causal direction statistical test for VARLiNGAM"""
    df = _stack(cause, effect, as_df=True)
    lagged = pd.concat([df.shift(1).add_suffix("_lag1"), df], axis=1).dropna()
    p = DirectLiNGAM().fit(lagged.values).get_error_independence_p_values(lagged.values)
    d_e, d_c = effect.shape[0], cause.shape[0]
    # True: causeₜ₋₁ → effectₜ
    p_true  = p[d_e : d_e + d_e, d_e - d_c : d_e].mean()
    # False: effectₜ₋₁ → causeₜ
    p_false = p[:d_c, :d_e].mean()
    return p_true, p_false


def conditional_tigramite(cause, effect):
    """Causal direction statistical test for PCMCI+"""
    df = pp.DataFrame(_stack(cause, effect), var_names=None)
    pm = PCMCI(df, ParCorr(significance="analytic"), verbosity=0).run_pcmciplus(tau_min=0, tau_max=1, pc_alpha=0.01)["p_matrix"]
    d_e, d_c = effect.shape[0], cause.shape[0]
    return pm[d_e:, :d_e, 1].mean(), pm[:d_e, d_e:, 1].mean()


def _rolling_window(cause, effect, window_length, callback, tau_max=1):
    """Compute rolling causal magnitude"""
    data = _stack(cause, effect)
    results = []
    for i in tqdm(range(len(data) - 1)):
        sub = center_slice(data, i, window_length)
        results.append(callback(sub))
    return normalize(torch.tensor(np.abs(results))).squeeze()


def magnitude_gvar(cause, effect, window_length=120, tau_max=1):
    """Causal magnitude evaluation: GVAR"""
    cb = lambda sub: training_procedure_trgc(
        data=sub,
        order=tau_max + 1,
        hidden_layer_size=16,
        end_epoch=50,
        batch_size=64,
        lmbd=0.0,
        gamma=0.0,
        verbose=False,
    )[1][: effect.shape[0], -cause.shape[0] :]
    return _rolling_window(cause, effect, window_length, cb, tau_max)


def magnitude_clstm(cause, effect, window_length=120, tau_max=1):
    """Causal magnitude evaluation: cLSTM"""
    def cb(sub):
        x = torch.tensor(sub, dtype=torch.float32).cuda().unsqueeze(0)
        model = cLSTM(x.shape[-1], hidden=16).cuda()
        train_model_ista(
            model,
            x,
            context=tau_max,
            lam=1e-2,
            lam_ridge=1e-2,
            lr=1e-1,
            max_iter=100,
            check_every=1,
            verbose=0,
        )
        return model.GC(threshold=False).detach().cpu().numpy()[
            : effect.shape[0], -cause.shape[0] :
        ]

    return _rolling_window(cause, effect, window_length, cb, tau_max)


def magnitude_tsci(cause, effect, window_length=120, tau_max=1):
    """Causal magnitude evaluation: TSCI"""
    def cb(sub):
        d_e = effect.shape[0]
        x, y = sub[:, :d_e], sub[:, d_e:]
        Qx = 2 # X embedding dim
        Qy = 2 # Y embedding dim
        xs = delay_embed(x, 1, Qx)
        ys = delay_embed(y, 1, Qy)
        dx = delay_embed(discrete_velocity(x), 1, Qx)
        dy = delay_embed(discrete_velocity(y), 1, Qy)
        return np.mean(tsci.tsci_nn(xs, ys, dx, dy, 0.9, False)[0])

    return _rolling_window(cause, effect, window_length, cb, tau_max)


def magnitude_varlingam(cause, effect, window_length=120, tau_max=1):
    """Causal magnitude evaluation: VARLiNGAM"""
    def cb(sub):
        df = pd.DataFrame(sub)
        lagged = pd.concat(
            [df.shift(lag).add_suffix(f"_lag{lag}") for lag in range(1, tau_max + 1)]
            + [df],
            axis=1,
        ).dropna()
        B = DirectLiNGAM().fit(lagged.values).adjacency_matrix_
        return B[: effect.shape[0], -cause.shape[0] :]

    return _rolling_window(cause, effect, window_length, cb, tau_max)


def magnitude_tigramite(cause, effect, window_length=120, tau_max=1):
    """Causal magnitude evaluation: PCMCI+"""
    def cb(sub):
        df = pp.DataFrame(sub, var_names=None)
        vm = PCMCI(df, ParCorr(significance="analytic"), verbosity=0).run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=0.01)["val_matrix"]
        return vm[: effect.shape[0], -cause.shape[0] :, tau_max]

    return _rolling_window(cause, effect, window_length, cb, tau_max)


def running_mean(x, window):
    """Compute running mean with given window size."""
    return np.convolve(np.asarray(x).squeeze(), np.ones(window) / window, mode='valid')


def detect_spans(rm, duration, high_thres=None, low_thres=None, high_label=None, low_label=None):
    """
    Return list of (start_idx, end_idx, label, color) for spans where
    rm stays above high_thres or below low_thres for `duration` steps.
    """
    spans, seen = [], set()
    for i in range(len(rm) - duration + 1):
        win = rm[i : i + duration]
        if high_thres is not None and all(win >= high_thres) and 'high' not in seen:
            spans.append((i, i+duration, high_label, 'C1'))
            seen.add('high')
        elif low_thres is not None and all(win <= low_thres) and 'low' not in seen:
            spans.append((i, i+duration, low_label, 'C0'))
            seen.add('low')
    return spans

def plot_anomalies_enso(ax, sst_anomalies, time_idx):
    """
    Plot El Niño and La Niña events with 5 consecutive 3-month running mean criterion.
    """
    
    elnino_threshold, lanina_threshold = 0.5, -0.5
    elnino_label, lanina_label = False, False

    # Calculate 3-month running mean
    running_mean = np.convolve(sst_anomalies, np.ones(3) / 3, mode='valid')

    # Check for 5 consecutive values exceeding the El Nino or La Nina threshold
    for i in range(len(running_mean) - 4):
        
        ## El Nino
        if all(running_mean[i:i + 5] >= elnino_threshold):
            ax.axvspan(
                time_idx[i + 2], time_idx[i + 3] if i + 5 < len(time_idx) else time_idx[-1],
                color='C1', alpha=0.2, label='El Niño' if not elnino_label else None
            )
            elnino_label = True


        ## La Nina
        elif all(running_mean[i:i + 5] <= lanina_threshold):
            ax.axvspan(
                time_idx[i + 2], time_idx[i + 3] if i + 5 < len(time_idx) else time_idx[-1],
                color='C0', alpha=0.2, label='La Niña' if not lanina_label else None
            )
            lanina_label = True

    ax.legend(loc='upper left', frameon=True)
    

def plot_anomalies(ax, X, time_idx, run_mean_window=3, event_duration=5, thres=1.0):
    """
    Plot anomaly events for a general time series X.
    """
    # Compute global mean and standard deviation
    X = np.array(X).squeeze()
    mean_val = np.mean(X)
    std_val = np.std(X)
    
    # Define thresholds as ±1 standard deviation from the mean.
    upper_threshold = mean_val + thres*std_val
    lower_threshold = mean_val - thres*std_val
    
    # Compute running mean
    running_mean = np.convolve(X, np.ones(run_mean_window) / run_mean_window, mode='valid')
    
    # Flags to only label each anomaly once
    high_label = False
    low_label = False
    
    # Loop over the running mean values
    for i in range(len(running_mean) - event_duration + 1):
        window = running_mean[i:i + event_duration]
        # High anomaly: running mean above upper_threshold
        if all(window >= upper_threshold):
            ax.axvspan(
                time_idx[i + run_mean_window - 1],
                time_idx[min(i + run_mean_window - 1 + event_duration, len(time_idx)-1)],
                color='C1', alpha=0.2,
                label = rf'$\bar{{\Omega}}_E + {thres}\sigma$' if not high_label else None
            )
            high_label = True
        
        # Low anomaly: running mean below lower_threshold
        elif all(window <= lower_threshold):
            ax.axvspan(
                time_idx[i + run_mean_window - 1],
                time_idx[min(i + run_mean_window - 1 + event_duration, len(time_idx)-1)],
                color='C0', alpha=0.2,
                label = rf'$\bar{{\Omega}}_E - {thres}\sigma$' if not low_label else None
            )
            low_label = True
    ax.legend(loc='upper left', frameon=True)

    
def get_anomalies_enso(sst_anomalies, time_idx, 
                       run_mean_window=3, event_duration=5, en_threshold=0.5, la_threshold=-0.5):
    """
    Returns a boolean array (or list) of the same length as time_idx,
    where True indicates that an ENSO event (El Niño or La Niña) was detected.
    """
    running_mean = np.convolve(sst_anomalies, np.ones(run_mean_window)/run_mean_window, mode='valid')
    enso_labels = np.zeros(len(time_idx), dtype=bool)
    
    for i in range(len(running_mean) - event_duration + 1):
        window = running_mean[i : i + event_duration]
        
        if all(window >= en_threshold) or all(window <= la_threshold):
            start_time_idx = i + run_mean_window - 1
            end_time_idx = i + event_duration + run_mean_window - 1
            end_time_idx = min(end_time_idx, len(time_idx))
            enso_labels[start_time_idx:end_time_idx] = True
    return enso_labels

def get_anomalies(X, time_idx, run_mean_window=3, event_duration=5, thres=1.0):
    """
    Returns a boolean array of the same length as time_idx, where True indicates that an anomaly
    event was detected based on the running mean of X being above or below ±thres*standard deviation.
    """
    X = np.array(X).squeeze()
    mean_val = np.mean(X)
    std_val = np.std(X)
    
    upper_threshold = mean_val + thres*std_val
    lower_threshold = mean_val - thres*std_val
    
    running_mean = np.convolve(X, np.ones(run_mean_window) / run_mean_window, mode='valid')
    anomalies = np.zeros(len(time_idx), dtype=bool)
    
    for i in range(len(running_mean) - event_duration + 1):
        window = running_mean[i:i + event_duration]
        if all(window >= upper_threshold) or all(window <= lower_threshold):
            start_idx = i + run_mean_window - 1
            end_idx = min(i + run_mean_window - 1 + event_duration, len(time_idx))
            anomalies[start_idx:end_idx] = True
    return anomalies
