import copy
import numpy as np
import torch

@torch.inference_mode()
def bootstrap_testing(samples):
    """
    Bootstrap significance summary for 'greater than 0'.

    Args:
        samples : tensor-like
            Bootstrap ensemble of your statistic. Shape (B,) or (B, ...).

    Returns:
        p_val        (One-sided test on bootstrap samples)
    """
    B = samples.shape[0]
    if B < 2:
        raise ValueError("Need at least 2 bootstrap samples.")

    # One-sided (>0) bootstrap p-value: Pr[stat <= 0]
    # Add-one smoothing to avoid 0 exactly.
    count_le0 = torch.sum(samples <= 0, axis=0)
    return (count_le0 + 1) / (B + 1)


@torch.inference_mode()
def hypothesis_testing(
    estimator,
    causal_effect,
    time_shift = 1,
    n_permutes = 30
):
    """
    Perform hypothesis testing comparing causal effect with randomized timeseries.
    Using empirical p-value computation.
    """
    # Random permutation
    non_causal_effect = torch.empty(n_permutes)
    
    for i in range(n_permutes):
        est_copy = copy.deepcopy(estimator)
        est_copy.effect = est_copy.effect[..., torch.randperm(est_copy.effect.size(-1))]
        est_copy.cause = est_copy.cause[..., torch.randperm(est_copy.cause.size(-1))]
        non_causal_effect[i] = est_copy.evaluate(time_shift=time_shift)

    # Compute empirical p-value (one-side)
    return ((non_causal_effect >= causal_effect).sum() + 1) / (non_causal_effect.numel() + 1)
