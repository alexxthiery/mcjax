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
from mcjax.proba.gaussian import IsotropicGauss,GMMFixed, MixedIsotropicGauss
from mcjax.proba.neal_funnel import NealFunnel
from mcjax.proba.student import Student
from mcjax.proba.banana2d import Banana2D
from mcjax.proba.doublewell import DoubleWell


print(f"Available devices: {jax.devices()}")
jax.config.update("jax_platform_name", "gpu")

METHOD_MAP = {'RWM': 0, 'MALA': 1}


# Functions to calculate the normalizing constant estimator
def single_run(GSMC, num_particles, key, mc_method_code, if_adaptive):
    if if_adaptive:
        _,_,log_weights_arr,_,_,coefs = GSMC.selfadaptive_run(num_particles, key, mc_method_code)
        return log_weights_arr,coefs
    else:
        _, log_weights_arr,_,_ = GSMC.run(num_particles, key, mc_method_code)
        return log_weights_arr, GSMC.coefs

def mult_run(GSMC:GeometricSMC, num_particles, key, mc_method, num_run, if_adaptive):
    keys = jr.split(key, num_run)
    batch_run = jax.vmap(single_run, in_axes=(None,None,0,None,None))
    batch_run = jax.jit(batch_run, static_argnums=(0,1,3,4))
    log_weight_batch, coefs_batch = batch_run(GSMC,num_particles,keys,METHOD_MAP[mc_method], if_adaptive)
    
    logZ_arr = [
        jnp.sum(logsumexp(log_weight_batch[i, :, 1:], axis=0)-jnp.log(num_particles))
        for i in range(num_run)
    ]
    # calculate the average coefs
    avg_coefs = jnp.mean(coefs_batch, axis=0)

    return jnp.array(logZ_arr), avg_coefs


def get_log_dist(name, param):
    dim, m, delta, sigma_x = param
    if name == 'funnel':
        return NealFunnel(dim=dim, sigma_x=sigma_x)
    elif name == 'gaussian': # Default reference: standard isotropic Gaussian
        mu_0 = jnp.zeros(dim)
        sigma_0 = 1.
        log_var_0 = jnp.log(sigma_0**2)
        return IsotropicGauss(mu=mu_0, log_var=log_var_0)
    elif name == 'student':
        mu = jnp.ones(dim)
        cov = jnp.eye(dim)
        deg = 3.
        return Student(mu=mu, cov=cov, deg=deg)
    elif name == 'banana2d':
        print(f"dim={dim}")
        assert dim == 2, "Banana2D is only defined for 2D"
        return Banana2D()
    elif name == 'gmmfixed':
        # 9-component Gaussian mixture model in 2D
        assert dim == 2, "GMMFixed is only defined for 2D"
        return GMMFixed()
    elif name == 'doublewell':
        return DoubleWell(dim=dim, m=m, delta=delta)
    elif name == 'mixedgaussian':
        # high-dimensional mixture of 2 Gaussians centered at (1,...,1) and (-1,...,-1) respectively
        mu1 = jnp.ones(dim)
        mu2 = -jnp.ones(dim)
        mu = jnp.stack([mu1, mu2], axis=0)
        log_var = jnp.log(jnp.ones(2))
        weights = jnp.array([0.5, 0.5])
        return MixedIsotropicGauss(mu=mu, log_var=log_var, weights=weights)
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
    parser.add_argument("--max_step", type=int, default=100)
    parser.add_argument("--ess_normalized_target", type=float, default=0.6)
    parser.add_argument("--if_adaptive", type=str2bool, default=True)
    parser.add_argument("--reference", type=str, default='Gaussian')
    parser.add_argument("--target", type=str, default='Funnel')
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--method", type=str, default='RWM', choices=['RWM', 'MALA'])

    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--delta", type=float, default=4.0)
    parser.add_argument("--sigma_x", type=float, default=3.0)

    return parser.parse_args()

# Test 1: Run SMC with different number of particles and estimate the normalizing constant
def smc_test1(): 
    args = parse_args()
    num_particles_arr = args.num_particles_arr
    num_run = args.num_runs
    step_size = args.step_size
    num_substeps = args.num_substeps
    max_step = args.max_step
    ess_normalized_target = args.ess_normalized_target
    if_adaptive = args.if_adaptive
    coefs = jnp.arange(args.num_steps+1)/args.num_steps
    reference = args.reference
    target = args.target
    method = args.method
    dim = args.dim

    param = (dim, args.m, args.delta, args.sigma_x)
    log_gamma_0 = get_log_dist(reference, param)
    log_gamma_T = get_log_dist(target, param)

    # print information of this test
    print(f"SMC Test 1: from {reference} to {target}, dim={dim}, method={method}, adaptive={if_adaptive}")

    # create a folder named 'pics' to save the plots
    if not os.path.exists('pics'):
        os.makedirs('pics')
    # create a subfolder for this target   
    target_folder = os.path.join('pics', target)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)


    key = jr.key(0)
    data1 = {"N_arr": num_particles_arr, "logZ": []}
    for num_particles in num_particles_arr:
        GSMC = GeometricSMC(log_gamma_0= log_gamma_0, log_gamma_T= log_gamma_T, coefs=coefs, \
                        step_size=step_size, num_substeps=num_substeps, keep_particles=False, \
                            max_step=max_step, ess_normalized_target=ess_normalized_target)
        print("Running with num_particles: ", num_particles)
        logZ_arr, avg_coefs = mult_run(GSMC, num_particles=num_particles, key=key, mc_method=method, \
                            num_run = num_run, if_adaptive=if_adaptive)
        data1["logZ"].append(logZ_arr)
    
    # convert all terms in avg_coefs to %.3f
    # avg_coefs = jnp.round(avg_coefs, 3)
    # print("Average coefs: ", avg_coefs.tolist())
    
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
    plt.savefig(os.path.join(target_folder, f'logZ_{reference}_{target}_DIM={dim}_{method}.png'))
    plt.close()

# Test 2: Compare adaptive vs non-adaptive SMC with RWM and MALA kernels
def smc_test2(): 
    args = parse_args()
    num_particles_arr = args.num_particles_arr
    num_run = args.num_runs
    step_size = args.step_size
    num_substeps = args.num_substeps
    max_step = args.max_step
    ess_normalized_target = args.ess_normalized_target
    coefs = jnp.arange(args.num_steps+1)/args.num_steps
    reference = args.reference
    target = args.target
    dim = args.dim

    param = (dim, args.m, args.delta, args.sigma_x)
    log_gamma_0 = get_log_dist(reference, param)
    log_gamma_T = get_log_dist(target, param)

    # print information of this test
    print(f"SMC Test 2: from {reference} to {target}, dim={dim}")

    # create a folder named 'pics' to save the plots
    if not os.path.exists('pics'):
        os.makedirs('pics')
    # create a subfolder for this target   
    target_folder = os.path.join('pics', target)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    key = jr.key(0)
    data1 = {"logZ_adaptive_RWM": [], \
             "logZ_nonadaptive_RWM": [], "logZ_adaptive_MALA": [], \
                "logZ_nonadaptive_MALA": []}
    for num_particles in num_particles_arr:
        GSMC = GeometricSMC(log_gamma_0= log_gamma_0, log_gamma_T= log_gamma_T, coefs=coefs, \
                        step_size=step_size, num_substeps=num_substeps, keep_particles=False, max_step=max_step, \
                            ess_normalized_target=ess_normalized_target)
        print("Running with num_particles: ", num_particles)
        logZ_arr,_ = mult_run(GSMC, num_particles=num_particles, key=key, mc_method='RWM', num_run = num_run, if_adaptive=True)
        data1["logZ_adaptive_RWM"].append(logZ_arr)
        logZ_arr,_ = mult_run(GSMC, num_particles=num_particles, key=key, mc_method='RWM', num_run = num_run, if_adaptive=False)
        data1["logZ_nonadaptive_RWM"].append(logZ_arr)
        logZ_arr,_ = mult_run(GSMC, num_particles=num_particles, key=key, mc_method='MALA', num_run = num_run, if_adaptive=True)
        data1["logZ_adaptive_MALA"].append(logZ_arr)
        logZ_arr,_ = mult_run(GSMC, num_particles=num_particles, key=key, mc_method='MALA', num_run = num_run, if_adaptive=False)
        data1["logZ_nonadaptive_MALA"].append(logZ_arr)
    
    # get the mean and var of each setting
    means_arr = []; vars_arr = []
    for key in data1.keys():
        means_arr.append([jnp.mean(data1[key][i]) for i in range(len(num_particles_arr))])
        vars_arr.append([jnp.var(data1[key][i]) for i in range(len(num_particles_arr))])

    
    # Plot the mean and variance of logZ estimates in 2 plots
    plt.figure(figsize=(12,5))
    positions = np.arange(len(num_particles_arr))
    plt.subplot(1,2,1)
    plt.plot(positions, means_arr[0], label='Adaptive RWM', marker='o')
    plt.plot(positions, means_arr[1], label='Non-adaptive RWM', marker='o')
    plt.plot(positions, means_arr[2], label='Adaptive MALA', marker='o')
    plt.plot(positions, means_arr[3], label='Non-adaptive MALA', marker='o')
    plt.axhline(y=log_gamma_T.log_Z() - log_gamma_0.log_Z(), color='r', linestyle='--', label='True logZ')
    plt.xticks(positions, num_particles_arr)
    plt.xlabel('Number of particles')
    plt.ylabel('Mean of logZ estimates')
    plt.title(f'Mean of logZ estimates ({reference} to {target})')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(positions, vars_arr[0], label='Adaptive RWM', marker='o')
    plt.plot(positions, vars_arr[1], label='Non-adaptive RWM', marker='o')  
    plt.plot(positions, vars_arr[2], label='Adaptive MALA', marker='o')
    plt.plot(positions, vars_arr[3], label='Non-adaptive MALA', marker='o')
    plt.xticks(positions, num_particles_arr)
    plt.xlabel('Number of particles')
    plt.ylabel('Variance of logZ estimates')
    plt.title(f'Variance of logZ estimates ({reference} to {target})')
    plt.legend()

    # create the directory if it does not exist
    if not os.path.exists('pics'):
        os.makedirs('pics')
    plt.savefig(os.path.join(target_folder, f'logZ_mean_var_{reference}_{target}_DIM={dim}.png'))
    plt.close()

if __name__ == "__main__":
    smc_test1()
    smc_test2()
