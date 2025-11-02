import abc
import torch
import copy
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from .regressors import PINV
from .observables import RandomFourierFeatures
from .koopman import Kausal
from .stats import bootstrap_testing

from .utils import get_logger
logger = get_logger()
    
class Graph(abc.ABC):
    """
    Wrapper class to Kausal to discovery full graph given pairwise tests.

    Parameters:
        regressor (BaseRegressor): Regressor object, defaults to PINV.
        marginal_observable (BaseObservables): Observables object for the marginal model, defaults to RFF.
        joint_observable (BaseObservables): Observables object for the joint model, defaults to RFF.

    Returns:
        self.
    """
    def __init__(
        self,
        regressor = PINV(),
        marginal_observable = RandomFourierFeatures(),
        joint_observable = RandomFourierFeatures()
    ):
        super().__init__()
        
        """Initialize Graph class"""
        super(Graph, self).__init__()
        
        self.regressor = regressor
        self.marginal_observable = marginal_observable
        self.joint_observable = joint_observable

    def infer(
        self,
        X,
        time_shift = 1,
        fit_kwargs = {},
        bootstrap_kwargs = {},
        **kwargs
    ):
        self.causal_graph = {}
        self.N = X.shape[0]
        pairs = [(i, j) for i in range(self.N) for j in range(self.N)]

        for i, j in pairs:
            cause = X[i]
            effect = X[j]

            # Initialize pairwise model
            marginal_observable = copy.deepcopy(self.marginal_observable)
            joint_observable = copy.deepcopy(self.joint_observable)
            model = Kausal(
                marginal_observable = marginal_observable,
                joint_observable = joint_observable,
                cause = cause,
                effect = effect,
            )

            # Fit observables if needed
            model.fit(**fit_kwargs)

            # Evaluate
            error, p_val = model.evaluate(time_shift=time_shift, **bootstrap_kwargs)
            error_mean = error.mean()
            error_std  = error.std(unbiased=True) if error.numel() > 1 else torch.tensor(0.)

            # Store
            self.causal_graph[(i, j)] = {
                'mean': error_mean.item(),
                'std':  error_std.item(),
                'pval': p_val.item()
            }

        # logger.info('Inference is done, view results by calling `print_result()`')


    def print_result(self):
        df = pd.DataFrame.from_dict(self.causal_graph, orient="index")
        df.index = [f"{i}->{j}" for (i, j) in df.index]
        print(df)
    
    def get_adjacency(
        self, 
        p_crit = 0.05
    ):
        adj = torch.zeros((self.N, self.N), dtype=torch.int)
        for (i, j), stats in self.causal_graph.items():
            if stats['pval'] < p_crit:
                adj[i, j] = 1
        return adj

    def plot_adjacency(self):
        adj = self.get_adjacency(p_crit=0.05)
        G = nx.from_numpy_array(adj.numpy(), create_using=nx.DiGraph)
        nx.draw(G, with_labels=True, arrows=True,  node_color="skyblue", node_size=1000, font_size=12)
        plt.show()
        
        