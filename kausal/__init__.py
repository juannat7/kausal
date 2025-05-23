from __future__ import annotations

from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

from .koopman import Kausal

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass


__all__ = [
    "observables",
    "regressors",
    "stats",
    "baselines",
    "Kausal"
]