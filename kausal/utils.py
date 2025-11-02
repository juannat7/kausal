import sys
import math
import logging
import torch
import numpy as np
import pandas as pd
from time import time as timer
from sklearn.metrics import average_precision_score, roc_auc_score

def numerical_rank_svd(K: torch.Tensor, eps=None):
    """
    Check rank deficiency for Koopman operators
    """
    K = K.to(torch.float64)
    if eps is None:
        eps = torch.finfo(K.dtype).eps  # ~2.22e-16 for float64

    K = K.to(dtype=torch.float64)
    S = torch.linalg.svdvals(K)
    tol = S.max() * max(K.shape) * eps
    rank = int((S > tol).sum().item())
    return rank


def get_logger(name: str = "Kausal", level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a logger object.

    Args:
        name: Name of the logger (use module name).
        level: Logging level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        A configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # prevent duplicate handlers in notebooks
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
    

def score(preds, labs, name="Result"):
    """
    Calculates AUROC and AUPRC metrics given preds and labs.
    Accepts either a 2D or a 3D tensor (batch of summary graphs).
    'name' is used for column naming in the output DataFrame.
    """
    # Some casting concerning input data type:
    if isinstance(preds, list):
        preds = np.array(preds)
    if isinstance(labs, list):
        labs = np.array(labs)
    if isinstance(preds, pd.DataFrame):
        preds = preds.values
    if isinstance(labs, pd.DataFrame):
        labs = labs.values

    # Expand dimensions if a single sample is provided.
    if preds.ndim == 2:
        preds = np.expand_dims(preds, 0)

    preds = np.array(preds)

    # Duplicate labels (assuming similar graph for batched timeseries)
    N, T, D = preds.shape
    labs = np.expand_dims(labs, 0)
    labs = np.repeat(labs, N, 0)
    labs = np.array(labs)

    # Individual scoring for each sample.
    auroc_ind = []
    auprc_ind = []
    for x in range(len(labs)):
        auroc_ind.append(
            roc_auc_score(y_true=labs[x].flatten(), y_score=preds[x].flatten())
        )
        auprc_ind.append(
            average_precision_score(
                y_true=labs[x].flatten(), y_score=preds[x].flatten()
            )
        )

    # Mean individual metrics (if any valid samples exist).
    auroc_ind = np.mean(auroc_ind) if len(auroc_ind) > 0 else float("nan")
    auprc_ind = np.mean(auprc_ind) if len(auprc_ind) > 0 else float("nan")

    # Joint calculation: flatten all samples.
    labs = labs.flatten()
    preds = preds.flatten()

    joint_auroc = roc_auc_score(labs, preds)
    joint_auprc = average_precision_score(labs, preds)
    null_model_auroc = roc_auc_score(labs, np.zeros_like(preds))
    null_model_auprc = average_precision_score(labs, np.zeros_like(preds))

    #  Joint SHD: total mismatches over the entire flattened batch
    joint_shd = np.sum(np.abs((preds >= 0.5).astype(int) - labs.astype(int)))

    out = pd.DataFrame(
        [
            joint_auroc,
            auroc_ind,
            null_model_auroc,
            joint_auprc,
            auprc_ind,
            null_model_auprc,
            joint_shd
        ],
        columns=[name],
        index=[
            "Joint AUROC",
            "Individual AUROC",
            "Null AUROC",
            "Joint AUPRC",
            "Individual AUPRC",
            "Null AUPRC",
            "Joint SHD"
        ],
    )
    out.index.name = "Metric"
    return out

def rmse(x, y):
    """Compute root mean-squared error"""
    return torch.sqrt(torch.mean((x - y)**2, dim=tuple(range(1, x.dim()))))

def normalize(x):
    """Min-max scaling"""
    return (x - x.min()) / (x.max() - x.min())

def validate(x): 
    """
    Flatten ND state/observables to valid 2D matrix with ND/T columns/rows.
    This is used to e.g., compute Koopman operator.
    """
    *ND, T = x.shape # T is the number of timesteps
    return x.reshape(math.prod(ND), T)
    