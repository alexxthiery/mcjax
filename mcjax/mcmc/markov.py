import abc
import jax
import jax.random as jr
#from dataclasses import dataclass
from flax import struct
from typing import Any


@struct.dataclass
class MCMCOutput:
    traj: Any     # Trajectory of the state 
    summary: Any  # Summary statistics of the trajectory


class MarkovKernel(abc.ABC):
    """ Abstract class for Markov Kernels:
    - init_state: initialize the state of the kernel
    - step: perform a single step of the kernel
    - summarize_stats_traj: summarize the statistics of the kernel trajectory
    """
    @abc.abstractmethod
    def init_state(self, x_init):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abc.abstractmethod
    def step(self, state, key):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abc.abstractmethod
    def summarize_stats_traj(self, stats_traj):
        raise NotImplementedError("Subclasses must implement this method")
    
    def run(self, key, n_samples, state_init):
        step_fn = self.step

        def scan_fn(carry, _):
            key, state = carry
            key, key_ = jr.split(key)
            state, stats = step_fn(state, key_)  # no closure
            return (key, state), (state, stats)
        
        key, key_ = jr.split(key)
        _, (state_traj, stats_traj) = jax.lax.scan(scan_fn, (key_, state_init), length=n_samples)
        stats_summary = self.summarize_stats_traj(stats_traj)
        
        # merge the state_trajectory and stats_summary
        # return {**state_traj, **stats_summary}
        return MCMCOutput(
            traj=state_traj,
            summary=stats_summary
        )
