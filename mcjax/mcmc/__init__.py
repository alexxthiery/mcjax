"""
MCMC module: Markov Chain Monte Carlo methods.

Kernels:
- rwm: Random Walk Metropolis
- mala: Metropolis-Adjusted Langevin Algorithm

Core:
- run_mcmc: Run a single MCMC chain
- run_mcmc_batch: Run multiple chains in parallel
- MCMCOutput: Container for chain trajectory and stats

Adaptation:
- adapt_step_size: Generic step-size adaptation
"""

from . import rwm
from . import mala
from .core import run_mcmc, run_mcmc_batch, MCMCOutput
from .adaptation import adapt_step_size

__all__ = [
    "rwm",
    "mala",
    "run_mcmc",
    "run_mcmc_batch",
    "MCMCOutput",
    "adapt_step_size",
]
