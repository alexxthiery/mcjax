import numpy as np  

import jax
import jax.numpy as jnp
import jax.random as jr
import pylab as plt

from typing import TypedDict
import time
from scipy.special import logsumexp

# add ../mcjax to the path
import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../../')

from mcjax.smc.geometric_smc import GeometricSMC
from mcjax.proba.gaussian import IsotropicGauss,GMM40,MixedIsotropicGauss
from mcjax.proba.neal_funnel import NealFunnel
from mcjax.proba.student import Student
from mcjax.proba.banana2d import Banana2D


print(f"Available devices: {jax.devices()}")
jax.config.update("jax_platform_name", "gpu")


# Functions to calculate the normalizing constant estimator
def single_run(GSMC:GeometricSMC, num_particles, key, mc_method):
    t, _, log_weight_arr,*_ = GSMC.selfadaptive_run(num_particles, key, mc_method)
    return t, log_weight_arr

def mult_run(GSMC:GeometricSMC, num_particles, key, mc_method, num_run):
    keys = jr.split(key, num_run)
    batch_run = jax.vmap(single_run, in_axes=(None,None,0,None))
    batch_run = jax.jit(batch_run, static_argnums=(0,1,3))
    t_arr, log_weight_batch = batch_run(GSMC,num_particles,keys,mc_method)
    
    logZ_arr = [
        jnp.sum(logsumexp(log_weight_batch[i, :, 1:], axis=0)-jnp.log(num_particles))
        for i, t in enumerate(t_arr)
    ]

    return jnp.array(logZ_arr)

def compute_variance(GSMC:GeometricSMC, key, num_particles, method,num_run):
    t_start = time.time()
    logZ_arr = mult_run(GSMC, num_particles, key, method, num_run)
    t_end = time.time() 
    print(f"Computation time (with jit): {t_end-t_start}")

    # Compute the variance of log(Z)
    variance = jnp.var(logZ_arr)
    mean = jnp.mean(logZ_arr)
    return mean, variance

def smc_test(log_gamma_0, log_gamma_T, num_particles_arr, key, method, target):
    dim = log_gamma_T._dim
    data1 = {"N_arr": num_particles_arr, "logZ": []}
    for num_particles in num_particles_arr:
        GSMC = GeometricSMC(log_gamma_0= log_gamma_0, log_gamma_T= log_gamma_T, coefs=coefs, \
                        step_size=1., num_substeps=jnp.max(jnp.array([10, dim//2])), keep_particles=False)
        print("Running with num_particles: ", num_particles)
        logZ_arr = mult_run(GSMC, num_particles=num_particles, key=key, mc_method=method, num_run = num_run)
        data1["logZ"].append(logZ_arr)
    # plot boxplot of logZ with confidence interval and mean
    plt.figure()
    positions = np.arange(len(num_particles_arr))
    diff = jnp.abs(data1["logZ"][-1] - (log_gamma_T._log_Z - log_gamma_0._log_Z))
    plt.boxplot(data1["logZ"], positions=positions, showmeans=True, meanline=True, notch=True, showfliers=False, whiskerprops=dict(color='orange'))

    # compare to logZ of funnel distribution
    plt.axhline(y=log_gamma_T._log_Z - log_gamma_0._log_Z, color='r', linestyle='--', label=f'True logZ (difference = {diff:.2f})')

    plt.xticks(positions, num_particles_arr)
    plt.xlabel('Number of particles')
    plt.ylabel('logZ')
    plt.title(f'Boxplot of logZ with respect to number of particles ({method})')
    plt.legend()
    # create the directory if it does not exist
    if not os.path.exists('pics'):
        os.makedirs('pics')
    plt.savefig(f'pics/logZ_{target}_DIM={dim}_{method}.png')
    plt.close()


key = jr.key(0)
num_particles_arr = [10000,30000, 50000]
num_run = 200
N = 10
coefs = jnp.arange(N+1)/N

# --------------------------- Test With Gaussian target ---------------------------
# dim = 2
# mu_0 = jnp.zeros(dim)
# sigma_0 = 1.
# log_var_0 = jnp.log(sigma_0**2)
# log_gamma_0 = IsotropicGauss(mu=mu_0, log_var=log_var_0)

# mu_1 = jnp.ones(dim)
# sigma_1 = 0.3
# log_var_1 = jnp.log(sigma_1**2)
# log_gamma_T = IsotropicGauss(mu=mu_1, log_var=log_var_1)
# key, key_ = jr.split(key)
# smc_test(log_gamma_0, log_gamma_T, num_particles_arr, key_, method='RWM', target='Gaussian')
# key, key_ = jr.split(key)
# smc_test(log_gamma_0, log_gamma_T, num_particles_arr, key_, method='MALA', target='Gaussian')

# dim = 10
# mu_0 = jnp.zeros(dim)
# sigma_0 = 1.
# log_var_0 = jnp.log(sigma_0**2)
# log_gamma_0 = IsotropicGauss(mu=mu_0, log_var=log_var_0)

# mu_1 = jnp.ones(dim)
# sigma_1 = 0.3
# log_var_1 = jnp.log(sigma_1**2)
# log_gamma_T = IsotropicGauss(mu=mu_1, log_var=log_var_1)
# key, key_ = jr.split(key)
# smc_test(log_gamma_0, log_gamma_T, num_particles_arr, key_, method='RWM', target='Gaussian')
# key, key_ = jr.split(key)
# smc_test(log_gamma_0, log_gamma_T, num_particles_arr, key_, method='MALA', target='Gaussian')

# dim = 50
# mu_0 = jnp.zeros(dim)
# sigma_0 = 1.
# log_var_0 = jnp.log(sigma_0**2)
# log_gamma_0 = IsotropicGauss(mu=mu_0, log_var=log_var_0)

# mu_1 = jnp.ones(dim)
# sigma_1 = 0.3
# log_var_1 = jnp.log(sigma_1**2)
# log_gamma_T = IsotropicGauss(mu=mu_1, log_var=log_var_1)
# key, key_ = jr.split(key)
# smc_test(log_gamma_0, log_gamma_T, num_particles_arr, key_, method='RWM', target='Gaussian')
# key, key_ = jr.split(key)
# smc_test(log_gamma_0, log_gamma_T, num_particles_arr, key_, method='MALA', target='Gaussian')

# dim = 100
# mu_0 = jnp.zeros(dim)
# sigma_0 = 1.
# log_var_0 = jnp.log(sigma_0**2)
# log_gamma_0 = IsotropicGauss(mu=mu_0, log_var=log_var_0)

# mu_1 = jnp.ones(dim)
# sigma_1 = 0.3
# log_var_1 = jnp.log(sigma_1**2)
# log_gamma_T = IsotropicGauss(mu=mu_1, log_var=log_var_1)
# key, key_ = jr.split(key)
# smc_test(log_gamma_0, log_gamma_T, num_particles_arr, key_, method='RWM', target='Gaussian')
# key, key_ = jr.split(key)
# smc_test(log_gamma_0, log_gamma_T, num_particles_arr, key_, method='MALA', target='Gaussian')

# --------------------------- Test With Funnel target ---------------------------
dim = 2
mu_0 = jnp.zeros(dim)
sigma_0 = 1.
log_var_0 = jnp.log(sigma_0**2)
# log_gamma_0 = IsotropicGauss(mu=mu_0, log_var=log_var_0)
# ------------------ Randomly select params for 5 components of mixed gaussian --------------------------
key, key1, key2, key3 = jr.split(key,4)
K = 40
mu = jax.random.normal(key1, shape=(K, dim))*10-10
dist_sigma = jax.random.uniform(
    key2,
    shape=(K,),
    minval=0.1,
    maxval=1.0
)
log_var = jnp.log(dist_sigma ** 2)
weights = jax.random.dirichlet(key3, alpha=jnp.ones((K,)))
log_gamma_0 = MixedIsotropicGauss(mu=mu, log_var=log_var, weights=weights)
log_gamma_T = NealFunnel(dim=dim)
key, key_ = jr.split(key)
smc_test(log_gamma_0, log_gamma_T, num_particles_arr, key_, method='RWM', target='Funnel')
key, key_ = jr.split(key)
smc_test(log_gamma_0, log_gamma_T, num_particles_arr, key_, method='MALA', target='Funnel')

# dim = 10
# mu_0 = jnp.zeros(dim)
# sigma_0 = 1.
# log_var_0 = jnp.log(sigma_0**2)
# # log_gamma_0 = IsotropicGauss(mu=mu_0, log_var=log_var_0) # one component
# # ------------------ Randomly select params for 5 components of mixed gaussian --------------------------
# key, key1, key2, key3 = jr.split(key,4)
# K = 20
# mu = jax.random.normal(key1, shape=(K, dim))*10-10
# dist_sigma = jax.random.uniform(
#     key2,
#     shape=(K,),
#     minval=0.1,
#     maxval=1.0
# )
# log_var = jnp.log(dist_sigma ** 2)
# weights = jax.random.dirichlet(key3, alpha=jnp.ones((K,)))
# log_gamma_0 = MixedIsotropicGauss(mu=mu, log_var=log_var, weights=weights)

# log_gamma_T = NealFunnel(dim=dim)
# key, key_ = jr.split(key)
# smc_test(log_gamma_0, log_gamma_T, num_particles_arr, key_, method='RWM', target='Funnel')
# key, key_ = jr.split(key)
# smc_test(log_gamma_0, log_gamma_T, num_particles_arr, key_, method='MALA', target='Funnel')

# dim = 50
# mu_0 = jnp.zeros(dim)
# sigma_0 = 1.
# log_var_0 = jnp.log(sigma_0**2)
# log_gamma_0 = IsotropicGauss(mu=mu_0, log_var=log_var_0)
# log_gamma_T = NealFunnel(dim=dim)
# key, key_ = jr.split(key)
# smc_test(log_gamma_0, log_gamma_T, num_particles_arr, key_, method='RWM', target='Funnel')
# key, key_ = jr.split(key)
# smc_test(log_gamma_0, log_gamma_T, num_particles_arr, key_, method='MALA', target='Funnel')

# dim = 100
# mu_0 = jnp.zeros(dim)
# sigma_0 = 1.
# log_var_0 = jnp.log(sigma_0**2)
# log_gamma_0 = IsotropicGauss(mu=mu_0, log_var=log_var_0)
# log_gamma_T = NealFunnel(dim=dim)
# key, key_ = jr.split(key)
# smc_test(log_gamma_0, log_gamma_T, num_particles_arr, key_, method='RWM', target='Funnel')
# key, key_ = jr.split(key)
# smc_test(log_gamma_0, log_gamma_T, num_particles_arr, key_, method='MALA', target='Funnel')

