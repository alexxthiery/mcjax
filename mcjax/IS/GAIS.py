import jax.numpy as jnp
import jax.random as jr
from mcjax.proba.density import LogDensity
from mcjax.proba.student import Student
from mcjax.proba.gaussian import Gauss
from mcjax.util.weights import ess_log_weight
from mcjax.util.log_sums import log_mean_exp_batch


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
        """
        Run the Generalized Adaptive Importance Sampling algorithm
        
        output:
        ------
        dictionary = {
            'samples':  # the final samples
            'weights':  # the final weights
            'mu':       # the final location
            'cov':      # the final covariance matrix
            'mu_traj':  # the trajectory of the location
            'cov_traj': # the trajectory of the covariance matrix
            'ess':      # the trajectory of the effective sample size
            }
        """
        # check that family is in ['gaussian', 'student']
        error_msg = "Family must be in ['gaussian', 'student']"
        assert family in ['gaussian', 'student'], error_msg
        
        # check dimension of mu_init and cov_init
        error_msg = "Invalid dimension of mu_init"
        assert len(mu_init) == self.dim, error_msg
        error_msg = "Invalid dimension of cov_init"
        assert cov_init.shape == (self.dim, self.dim), error_msg
        
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
        
        # to store the individual proposal densities:
        # log_prop_indiv contains log pi_i(x_j) for all i and j
        log_prop_indiv = []
            
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
                'student': Student(mu=mu_approx, cov=cov_approx, deg=deg),
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
            
            # compute all the "deterministic mixtures weights"
            # OLD:: log_prop_indiv = [jnp.concatenate([prop.batch(x_)[:, None] for prop in prop_list], axis=1) for x_ in samples]
            log_prop_indiv = [jnp.concatenate([log_p_arr, dist.batch(x_)[:, None]], axis=1) for (log_p_arr, x_) in zip(log_prop_indiv, samples[:-1])]
            log_prop_indiv.append(jnp.concatenate([prop.batch(x)[:, None] for prop in prop_list], axis=1))
            
            log_prop = [log_mean_exp_batch(log_w) for log_w in log_prop_indiv]
            # log_weights = [self.logtarget.batch(x) - log_q for (x, log_q) in zip(samples, log_prop)]
            log_weights = [log_t-log_q for (log_t, log_q) in zip(log_target_values, log_prop)]
            
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
            cov_approx = jnp.cov(samples_all,
                                 rowvar=False,
                                 aweights=weights_all)    
            
            # TODO: add effective sample size check
            # TODO: check that the covariance matrix is positive definite
                    
        dict_output = {
            'samples': samples_all,     # the final samples
            'weights': weights_all,     # the final weights
            'mu': mu_params[-1],        # the final location
            'cov': cov_params[-1],      # the final covariance matrix
            'mu_traj': mu_params,       # the trajectory of the location
            'cov_traj': cov_params,     # the trajectory of the covariance matrix
            'ess': ess_list,            # the trajectory of the effective sample size
        }
        return dict_output

