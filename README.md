Kausal
=========

🚧 **Kausal** is a Python package to perform causal inference of nonlinear, high-dimensional dynamics using deep Koopman operator.

# Quickstart Guide
> __NOTE__: Please refer to our example notebooks (in `examples/`) for demonstration!


`Kausal` provides an interface to perform causal analysis based on Koopman operator theory, between two set of multivariate timeseries:

```python
from kausal import koopman

# Define your cause/effect variables
cause = ...  # Shape (n_features, *, n_timestep)
effect = ... # Shape (n_features, *, n_timestep)

# Initialize `Kausal` object
model = koopman.Kausal(
    cause = cause,
    effect = effect
)

# Estimate causal effect
causal_effect = model.evaluate(
    time_shift = 1
)
```

You can also specify specific observable functions, e.g., `MLP` or `CNN`.
```python
from kausal import koopman
from kausal.observables import MLPFeatures

model = koopman.Kausal(
    marginal_observable = MLPFeatures(...),
    joint_observable = MLPFeatures(...),
    cause = ...,
    effect = ...
)
```

Several regression techniques to estimate the Koopman operator is also provided, e.g., pseudo-inverse (`PINV`) or low-rank dynamic mode decomposition (`DMD`).
```python
from kausal import koopman
from kausal.regressors import DMD

model = koopman.Kausal(
    regressor = DMD(svd_rank = 4),
    cause = ...,
    effect = ...
)
```

