import jax
import equinox as eqx

def count_params(model):
    """Counts the number of trainable parameters."""
    trainable_params = eqx.filter(model, eqx.is_inexact_array)
    return sum(param.size for param in jax.tree_util.tree_leaves(trainable_params))

def normalize(x, mean, std):
    """Perform z-normalization"""
    return (x - mean) / std

def denormalize(x, mean, std):
    """Perform denormalization"""
    return (x * std) + mean

def flatten(x): 
    """Flatten N-D to matrix with ND/T columns/rows"""
    return x.reshape(-1, x.shape[-1])