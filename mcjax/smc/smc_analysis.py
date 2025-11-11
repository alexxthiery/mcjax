import numpy as np  

import jax
import jax.numpy as jnp
import jax.random as jr
import pylab as plt

import argparse
from typing import TypedDict
import time
from scipy.special import logsumexp

# add ../mcjax to the path
import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../../')

from mcjax.smc.geometric_smc import GeometricSMC
from mcjax.proba.gaussian import IsotropicGauss,GMMFixed
from mcjax.proba.neal_funnel import NealFunnel
from mcjax.proba.student import Student
from mcjax.proba.banana2d import Banana2D


print(f"Available devices: {jax.devices()}")
jax.config.update("jax_platform_name", "gpu")

METHOD_MAP = {'RWM': 0, 'MALA': 1}


# Functions to calculate the normalizing constant estimator
def single_run(GSMC, num_particles, key, mc_method_code, if_adaptive):
    if if_adaptive:
        return GSMC.selfadaptive_run(num_particles, key, mc_method_code)[2]
    else:
        return GSMC.run(num_particles, key, mc_method_code)[1]

def mult_run(GSMC:GeometricSMC, num_particles, key, mc_method, num_run, if_adaptive):
    keys = jr.split(key, num_run)
    batch_run = jax.vmap(single_run, in_axes=(None,None,0,None,None))
    batch_run = jax.jit(batch_run, static_argnums=(0,1,3,4))
    log_weight_batch = batch_run(GSMC,num_particles,keys,METHOD_MAP[mc_method], if_adaptive)
    
    logZ_arr = [
        jnp.sum(logsumexp(log_weight_batch[i, :, 1:], axis=0)-jnp.log(num_particles))
        for i in range(num_run)
    ]

    return jnp.array(logZ_arr)


def get_log_dist(name, dim):
    if name == 'funnel':
        return NealFunnel(dim=dim)
    elif name == 'gaussian':
        mu_0 = jnp.zeros(dim)
        sigma_0 = 1.
        log_var_0 = jnp.log(sigma_0**2)
        return IsotropicGauss(mu=mu_0, log_var=log_var_0)
    elif name == 'student':
        df = 3
        mu = jnp.zeros(dim)
        scale = 2.
        return Student(df=df, mu=mu, scale=scale)
    elif name == 'banana2d':
        assert dim == 2, "Banana2D is only defined for 2D"
        return Banana2D()
    elif name == 'gmmfixed':
        assert dim == 2, "GMMFixed is only defined for 2D"
        return GMMFixed()
    else:
        raise ValueError(f"Unknown distribution name: {name}")


def parse_args():
    def str2bool(v):
        return v.lower() in ('true', '1', 'yes')
    parser = argparse.ArgumentParser(description="Neural Sampler Experiments")
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--num_runs", type=int, default=200)
    parser.add_argument("--num_particles_arr", nargs='+', type=int, default=[100, 1000, 5000, 10000])
    parser.add_argument("--step_size", type=float, default=1.0)
    parser.add_argument("--num_substeps", type=int, default=10)
    parser.add_argument("--if_adaptive", type=str2bool, default=True)
    parser.add_argument("--reference", type=str, default='Gaussian')
    parser.add_argument("--target", type=str, default='Funnel')
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--method", type=str, default='RWM', choices=['RWM', 'MALA'])

    return parser.parse_args()

def smc_test1(): 
    args = parse_args()
    num_particles_arr = args.num_particles_arr
    num_run = args.num_runs
    step_size = args.step_size
    num_substeps = args.num_substeps
    if_adaptive = args.if_adaptive
    coefs = jnp.arange(args.num_steps+1)/args.num_steps
    reference = args.reference
    target = args.target
    method = args.method
    dim = args.dim
    log_gamma_0 = get_log_dist(reference, dim)
    log_gamma_T = get_log_dist(target, dim)

    key = jr.key(0)
    data1 = {"N_arr": num_particles_arr, "logZ": []}
    for num_particles in num_particles_arr:
        GSMC = GeometricSMC(log_gamma_0= log_gamma_0, log_gamma_T= log_gamma_T, coefs=coefs, \
                        step_size=step_size, num_substeps=num_substeps, keep_particles=False)
        print("Running with num_particles: ", num_particles)
        logZ_arr = mult_run(GSMC, num_particles=num_particles, key=key, mc_method=method, num_run = num_run, if_adaptive=if_adaptive)
        data1["logZ"].append(logZ_arr)
    # plot boxplot of logZ with confidence interval and mean
    plt.figure()
    positions = np.arange(len(num_particles_arr))
    diff = jnp.abs(jnp.mean(data1["logZ"][-1]) - (log_gamma_T.log_Z() - log_gamma_0.log_Z()))
    plt.boxplot(data1["logZ"], positions=positions, showmeans=True, meanline=True, notch=True, showfliers=False, whiskerprops=dict(color='orange'))

    # compare to logZ of funnel distribution
    plt.axhline(y=log_gamma_T.log_Z() - log_gamma_0.log_Z(), color='r', linestyle='--', label=f'True logZ (difference = {diff:.2f})')
    plt.xticks(positions, num_particles_arr)
    plt.xlabel('Number of particles')
    plt.ylabel('logZ')
    plt.title(f'Boxplot of logZ with respect to number of particles ({method})')
    # print the parameter settings
    plt.text(0.5, 0.9, f'Steps: {args.num_steps}\n Substeps: {num_substeps}\n Step size: {step_size}\n Adaptive: {if_adaptive}', 
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.legend()
    # create the directory if it does not exist
    if not os.path.exists('pics'):
        os.makedirs('pics')
    plt.savefig(f'pics/logZ_{reference}_{target}_DIM={dim}_{method}.png')
    plt.close()


if __name__ == "__main__":
    smc_test1()
