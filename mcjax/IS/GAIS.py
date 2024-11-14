import jax
import jax.numpy as jnp
import jax.random as jr
from typing import TypedDict, Tuple, Dict
from mcjax.proba.density import LogDensity
from mcjax.proba.student import Student
from mcjax.proba.gaussian import Gauss
from mcjax.util.ess import ess_log_weight


# ==========================================
# Generalized Adaptive Importance Sampling
#
# Reference:
# 1: Adaptive multiple importance sampling.
# Cornuet, J.M., Marin, J.M., Mira, A. and Robert, C.P., 2012. 
# Scandinavian Journal of Statistics, 39(4), pp.798-812.
# ==========================================
class GAIS:
    """ Generalized Adaptive Importance Sampling """
    def __init__(
                self,
                *,
                logtarget: LogDensity,  # logtarget distribution
                ):
        self.logtarget = logtarget
        self.dim = logtarget.dim
        
    def log_mean_exp_batch(
                self,
                x_arr: jnp.ndarray,  # (N, D): N number of samples, D dimension
            ):
        """ log mean exp implemented in a stable way """
        # max of each row
        x_max = jnp.max(x_arr, axis=1)
        # subtract the max for stability
        x_arr_normalized = x_arr - x_max[:, None]
        # compute the log sum exp
        return x_max + jnp.log(jnp.mean(jnp.exp(x_arr_normalized), axis=1))
    
    def log_sum_exp(
                self,
                x: jnp.ndarray,
            ):
        """ log sum exp implemented in a stable way """
        # max of each row
        x_max = jnp.max(x)
        # subtract the max for stability
        x_normalized = x - x_max
        # compute the log sum exp
        return x_max + jnp.log(jnp.sum(jnp.exp(x_normalized)))
    
    def run(self,
            *,
            key: jnp.ndarray,               # random key
            n_samples: int,                 # number of samples
            n_iter: int,                    # number of iterations
            mu_init: jnp.ndarray,           # initial location
            cov_init: jnp.ndarray,          # initial covariance
            family: str = 'gaussian',       # proposal family
            deg: int = 3,                   # degrees of freedom for Student-t proposal
            verbose: bool = False,          # verbose
            ):
        # check that family is in ['gaussian', 'student']
        error_msg = "Family must be in ['gaussian', 'student']"
        assert family in ['gaussian', 'student'], error_msg
        
        # check dimension of mu_init and cov_init
        error_msg = "Invalid dimension of mu_init"
        assert len(mu_init) == self.dim, error_msg
        error_msg = "Invalid dimension of cov_init"
        assert cov_init.shape == (self.dim, self.dim), error_msg
        
        # degree of freedom for the Student-t proposal if family is 'student'
        self.deg = deg
        
        mu_approx = mu_init
        cov_approx = cov_init
        
        # to store all the samples
        samples = []
        
        # to store the log_target values
        log_target_values = []
        
        # to store all the parameters
        mu_params = []
        cov_params = []
        
        # save the list of proposal densities
        prop_list = []

        # effective sample size
        ess_list = []
            
        for it in range(n_iter):
            if verbose:
                print(f"Iteration {it+1}/{n_iter}")
                
            # save the parameters
            mu_params.append(mu_approx)
            cov_params.append(cov_approx)
            
            # set the proposal distribution
            # remark: not optimal since both Gauss and Student
            # have to be initialized at each iteration, with potential
            # inversion / factorization of the covariance matrix
            # but fine for the moment
            dist_map = {
                'gaussian': Gauss(mu=mu_approx, cov=cov_approx),
                'student': Student(mu=mu_approx, cov=cov_approx, deg=self.deg),
            }
            dist = dist_map[family]

            # save the proposal distribution
            prop_list.append(dist)
            
            # sample from the proposal distribution
            key, key_ = jr.split(key)
            x = dist.sample(key=key_, n_samples=n_samples)
            
            # save the samples
            samples.append(x)
            
            # update the log_target values
            log_target_values.append(self.logtarget.batch(x))            
            
            # TODO: there are many duplicated computations here: proposals are re-computed and do not need to be
            # compute all the "deterministic mixtures weights"
            log_prop_indiv = [jnp.concatenate([prop.batch(x_)[:, None] for prop in prop_list], axis=1) for x_ in samples]
            log_prop = [self.log_mean_exp_batch(log_w) for log_w in log_prop_indiv]
            # log_weights = [self.logtarget.batch(x) - log_q for (x, log_q) in zip(samples, log_prop)]
            log_weights = [log_t - log_q for (log_t, log_q) in zip(log_target_values, log_prop)]
            

            # flatten everything
            log_weights_all = jnp.concatenate(log_weights)
            log_weights_all = log_weights_all - jnp.max(log_weights_all)
            samples_all = jnp.concatenate(samples)
            
            # compute the gaussianized weights
            weights_all = jnp.exp(log_weights_all)
            weights_all = weights_all / jnp.sum(weights_all)
            
            # compute the effective sample size and save it
            ess_val = ess_log_weight(log_weights_all)
            ess_list.append(ess_val)
            
            # update the proposal distribution
            mu_approx = jnp.sum(weights_all[:, None] * samples_all, axis=0)
            cov_approx = jnp.cov(samples_all, rowvar=False, aweights=weights_all)    
            
            # TODO: add effective sample size check
            # TODO: check that the covariance matrix is positive definite
                    
        dict_output = {
            'samples': samples_all,
            'weights': weights_all,
            'mu': mu_params[-1],
            'cov': cov_params[-1],
            'mu_traj': mu_params,
            'cov_traj': cov_params,
            'ess': ess_list,
        }
        
        return dict_output
    
