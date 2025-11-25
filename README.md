Kausal: Deep Koopman Operators for Causal Discovery
=========

**Kausal** is a PyTorch package to perform causal inference in nonlinear, high-dimensional dynamics using deep Koopman operator-theoretic formalism.

<div align="center">

  <!-- Python -->
  <a href="#">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white" alt="Python"/>
  </a>

  <!-- PyTorch -->
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch"/>
  </a>

  <!-- License -->
  <a href="https://github.com/juannat7/kausal/blob/main/LICENSE.txt">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/>
  </a>

  <img src="https://img.shields.io/pypi/dm/kausal"/>

</div>
<br/>

# Features
- 🌡️ [Causal measures with uncertainty quantification](#Causal-estimation)
- ⌛ [Causal emulation](#Causal-emulation)
- 🌐 [Causal graph discovery](#Causal-graph-discovery)

Additional features include:
- 🤖 [Using deep learning](#Using-deep-learning)
- 📶 [Low rank decomposition](#Using-low-rank)



# Abstract

![Overview of Kausal](docs/schematic-algorithm.png)
Causal discovery aims to identify cause-effect mechanisms for better scientific understanding, explainable decision-making, and more accurate modeling. Standard statistical frameworks, such as Granger causality, lack the ability to quantify causal relationships in nonlinear dynamics due to the presence of complex feedback mechanisms, timescale mixing, and nonstationarity. Thus, applying these methods to study causal dynamics in real-world systems, such as the Earth, is a major challenge. Addressing this shortcoming, we leverage deep learning and a **K**oopman operator-theoretic formalism to present a new class of c**ausal** discovery algorithms. **Kausal** uses deep Koopman operator methods to approximate nonlinear dynamics in a linearized vector space in which traditional causal inference methods such as Granger causality can be more easily applied. Our idealized experiments demonstrate **Kausal**'s superior ability in discovering and characterizing causal signals compared to existing deep learning and non-deep learning state-of-the-art approaches. Finally, the successful identification of major El Niño and La Niña events in observations showcases **Kausal**'s skill to handle real-world applications.

# Installation

Kausal is available on PyPi, so installation is as easy as:

```bash
pip install kausal
```

If you use conda, please use the following commands:
```bash
conda create --name venv python=3.10
conda activate venv
pip install kausal
```

# Quickstart Guide

Please refer to our tutorial notebooks in the `tutorial/` folder for full demonstration.

## Causal estimation
The most basic functionality is to perform causal estimation useful for e.g., event detection, relative strength measurements between variables.
```python
import torch
from kausal.koopman import Kausal

# Define cause-effect variables to be tested.
x_cause = torch.randn(3, 1000) # (n_channels, n_timesteps)
x_effect = torch.randn(3, 1000) # (n_channels, n_timesteps)

# Initialize the Kausal object
causal_koopman = Kausal(cause = x_cause, effect = x_effect)

# Evaluate (with e.g., time_shift = 1)
causal_effect, p_values = causal_koopman.evaluate(
    time_shift=1, 
    bootstrap_ratio=0.9, ## Subtrajectory length for uncertainty quantification
    bootstrap_nums=100   ## Number of resampling for uncertainty quantification
)
```

## Causal emulation
Once you fit your Koopman operators under some time shift, you can perform rollouts.
```python
import torch
from kausal.koopman import Kausal

# Define cause-effect variables to be tested.
x_cause = torch.randn(3, 1000) # (n_channels, n_timesteps)
x_effect = torch.randn(3, 1000) # (n_channels, n_timesteps)

# Initialize the Kausal object
causal_koopman = Kausal(cause = x_cause, effect = x_effect)

# Evaluate (with e.g., time_shift = 1)
x_forecast_marginal, x_forecast_joint = causal_koopman.forecast(
    n_train = int(0.8 * 1000), # Number of time samples used for training
    time_shift = 1
)
```

## Causal graph discovery
Ultimately, we can iterate through pairwise combination of variables to deduce their overall causal structures. 

```python
import torch
from kausal import Graph

# Define cause-effect variables to be tested.
x = torch.randn(10, 3, 1000) # (n_vars, n_channels, n_timesteps)

# Initialize Graph object
graph_model = Graph()

# Evaluate
graph_model.infer(
    X = x,
    time_shift = 100,
    bootstrap_kwargs = {'bootstrap_ratio': 0.9, 'bootstrap_nums': 30}
)

# Get some results
graph_model.get_adjacency() # Print out graph adjacency
graph_model.print_result()  # Print out p_values, causal measures and its uncertainties
```

# Advanced Guides
## Using deep learning
You can use deep learning-based features for the observables.
```python
import torch
from kausal.koopman import Kausal
from kausal.observables import MLPFeatures

# Define cause-effect variables to be tested.
x_cause = torch.randn(3, 1000) # (n_channels, n_timesteps)
x_effect = torch.randn(3, 1000) # (n_channels, n_timesteps)

# Initialize Kausal object (note the extra observables parameters)
causal_koopman = Kausal(
    marginal_observable = MLPFeatures(in_channels=3, hidden_channels=hidden_channels, out_channels=3),
    joint_observable = MLPFeatures(in_channels=6, hidden_channels=hidden_channels, out_channels=3),
    cause = x_cause,
    effect = x_effect,
)

# Fit the observables
marginal_loss_ce, joint_loss_ce = causal_koopman.fit(
    n_train = int(0.8 * 1000), 
    epochs = 500, 
    lr = 1e-2, 
    batch_size = int(0.8 * 1000)
)

# Evaluate (with e.g., time_shift = 1)
causal_effect, p_values = causal_koopman.evaluate(time_shift=1)
```
## Using low-rank
Low-rank estimators are also available e.g., through SVD.

```python
import torch
from kausal.koopman import Kausal
from kausal.regressors import DMD

# Initialize Kausal object
model = Kausal(
    regressor = DMD(svd_rank = 4),
    cause = torch.tensor(...),
    effect = torch.tensor(...)
)
```

# Experimental Results
You can find accompanying code to reproduce the experimental results in the `experiments/` folder.

# Developer's Guide
We welcome and appreciate any contribution to improve the codebase! You can make a Pull Request or raise an Issue. During development, install the package in the editable format:

```bash
git clone https://github.com/juannat7/kausal.git
cd kausal/
pip install -e .
```

# Citation
If you find any of the code and dataset useful, feel free to acknowledge our work through:

```bibtex
@article{nathaniel2025deepkoopmanoperatorframework,
  title={Deep Koopman operator framework for causal discovery in nonlinear dynamical systems},
  author={Juan Nathaniel and Carla Roesch and Jatan Buch and Derek DeSantis and Adam Rupe and Kara Lamb and Pierre Gentine},
  journal={arXiv preprint arXiv:2505.14828},
  year={2025}
}

@article{rupe2024causal,
  title={Causal Discovery in Nonlinear Dynamical Systems using Koopman Operators},
  author={Rupe, Adam and DeSantis, Derek and Bakker, Craig and Kooloth, Parvathi and Lu, Jian},
  journal={arXiv preprint arXiv:2410.10103},
  year={2024}
}
