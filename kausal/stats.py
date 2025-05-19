import copy
import torch

def hypothesis_testing(
    estimator,
    time_shift = 1,
    n_permutes = 30
):
    """
    Perform hypothesis testing comparing causal effect with jumbled up timeseries.
    Using empirical p-value computation.
    """
    causal_effect = estimator.evaluate(time_shift = time_shift)

    # Random permutation (copy instantiation)
    non_causal_effect = torch.empty(n_permutes)
    for i in range(n_permutes):
        est_copy = copy.deepcopy(estimator)
        est_copy.effect = est_copy.effect[..., torch.randperm(est_copy.effect.size(-1))]
        est_copy.cause = est_copy.cause[..., torch.randperm(est_copy.cause.size(-1))]
        non_causal_effect[i] = est_copy.evaluate(time_shift = time_shift)

    # Compute empirical p-value (one-side)
    return ((non_causal_effect >= causal_effect).sum() + 1) / (non_causal_effect.numel() + 1)
        