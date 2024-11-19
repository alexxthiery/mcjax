import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy
from mcjax.proba.density import LogDensity
from mcjax.proba.student import Student
from mcjax.proba.gaussian import Gauss
from mcjax.util.weights import ess_log_weight, target_ess_normalized
from mcjax.util.weights import compute_weights
import mcjax.util.psd as psd


# ==========================================
# Doubly Adaptive Importance Sampling
# Reference:
# "Doubly Adaptive Importance Sampling"  
# van den Boom, W. Thiery, A.H., and Cremaschi, A (2023)  
# arxiv: https://arxiv.org/abs/2404.18556
# ==========================================
class DAIS:
    """ Doubly Adaptive Importance Sampling
    
    Reference:
    ----------
    "Doubly Adaptive Importance Sampling"  
    van den Boom, W. Thiery, A.H., and Cremaschi, A (2023)  
    arxiv: https://arxiv.org/abs/2404.18556
    """
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
            ess_threshold: float = 0.5,     # effective sample size threshold
            alpha_damp: float = 1.,         # damping factor for the mean/covariance updates
            verbose: bool = False,          # verbose
            ):
        """
        Reference:
        ----------
        "Doubly Adaptive Importance Sampling"  
        van den Boom, W. Thiery, A.H., and Cremaschi, A (2023)  
        arxiv: https://arxiv.org/abs/2404.18556

        output:
        =======

        dict = {
            "mu": mu,                       # mu parameter
            "cov": cov,                     # covariance parameter
            "elbo": elbo_list[-1],          # elbo value
            "mu_traj": mu_params,           # mu param trajectory
            "cov_traj": cov_params,         # cov param
            "elbo_traj": elbo_list,         # elbo trajectory
            "eps_traj": eps_params,         # eps trajectory
            "mean_hat_traj": mean_hat_list,   # estimation of mean (**different from param mu**)
            "cov_hat_traj": cov_hat_list,   # estimation of covariance (**different from param cov**)
        }
        """
        # check dimension of mu_init and cov_init
        error_msg = "cov_invalid dimension of mu_init"
        assert len(mu_init) == self.dim, error_msg
        error_msg = "cov_invalid dimension of cov_init"
        assert cov_init.shape == (self.dim, self.dim), error_msg
                
        # initialize
        cov = jnp.copy(cov_init)
        mu = jnp.copy(mu_init)

        # list to save history
        elbo_list = []
        ess_normalized_list = []
        mu_params = []
        cov_params = []
        eps_params = []
        mean_hat_list = []
        cov_hat_list = []
        
        # initialize the proposal distribution
        proposal = Gauss(mu=mu, cov=cov)

        @jax.jit
        def compute_logtarget(xs):
            return self.logtarget.batch(xs)
        
        @jax.jit
        def compute_logtarget_grad(xs):
            return self.logtarget.grad_batch(xs)
        
        for it in range(n_iter):
            # generate samples from current approx
            key, key_ = jr.split(key)
            xs = proposal.sample(key=key_, n_samples=n_samples, center=True)

            # compute importance weights
            # log_pi = self.logtarget.batch(xs)
            log_pi = compute_logtarget(xs)
            log_q = proposal.batch(xs)
            # log_q = compute_logproposal(xs)
            log_w = log_pi - log_q
            
            # compute ess normalized
            ess_normalized = ess_log_weight(log_w) / n_samples
            
            # compute elbo
            elbo = jnp.mean(log_w)
            
            # estimate mean / variance
            prop_student = Student(mu=mu, cov=cov, deg=3)
            key, key_ = jr.split(key)
            xs_student = prop_student.sample(key=key_, n_samples=n_samples, center=True)
            log_q_student = prop_student.batch(xs_student)
            # log_pi_student = self.logtarget.batch(xs_student)
            log_pi_student = compute_logtarget(xs_student)
            log_w_student = log_pi_student - log_q_student
            w_student_normalized = compute_weights(log_w_student)
            mean_student_hat = jnp.sum(w_student_normalized[:, None] * xs_student, axis=0)
            xs_student_centred = xs_student - mean_student_hat[None, :]
            cov_student_hat = xs_student_centred.T @ (w_student_normalized[:, None] * xs_student_centred)
            
            # find the next temperature
            eps = target_ess_normalized(
                        log_weights=log_w,
                        ess_normalized_target=ess_threshold)

            # grad_phi = grad_log(target) - grad_log_gaussian
            # logtarget_grad = self.logtarget.grad_batch(xs)
            logtarget_grad = compute_logtarget_grad(xs)
            grad_phis = logtarget_grad - proposal.grad_batch(xs)
            cov_grad_phis = grad_phis @ cov.T
            
            while True:
                # compute normalized weights
                w_tempered = compute_weights(eps*log_w)
                                
                # update the mean:
                # ===========
                # mu_new = mu + eps * E_{tempered}[cov @ grad_phi]
                mean_cov_grad_phis = jnp.sum(w_tempered[:, None]*cov_grad_phis, axis=0)
                mu_new = mu + eps * mean_cov_grad_phis
                
                # update the covariance
                # =================
                # cov_new = cov + eps * E_{tempered}[Cov(cov@grad_phi, x)]
                cov_grad_phis_centred = cov_grad_phis - mean_cov_grad_phis[None, :]
                xs_mean = jnp.sum(w_tempered[:, None] * xs, axis=0)
                xs_centred = xs - xs_mean[None, :]
                delta_cov = cov_grad_phis_centred.T @ (w_tempered[:, None] * xs_centred)
                # make sure that the covariance matrix is symmetric
                delta_cov = 0.5*(delta_cov+delta_cov.T)
                cov_new = cov + eps * delta_cov
                
                # sanity checks
                # =============
                if psd.is_psd(cov + alpha_damp*(cov_new-cov)):
                    break
                else:
                    # if the covariance matrix is not psd, reduce the temperature
                    if verbose:
                        print(f"\t Reducing the temperature: eps={eps:.2f}")
                    eps = 0.5*eps
            
            # save the params
            mu_params.append(mu)
            cov_params.append(cov)
            eps_params.append(eps)
            elbo_list.append(elbo)
            ess_normalized_list.append(ess_normalized)
            
            # save estimated mean and covariance
            mean_hat_list.append(mean_student_hat)
            cov_hat_list.append(cov_student_hat)
            
            # print information to the user
            if verbose:
                print(f"[{it+1}/{n_iter}] \t eps: {eps:.2f} \t ELBO: {elbo:.2f} \t ESS: {ess_normalized:.3f}")

            # update the parameters
            mu = mu + alpha_damp*(mu_new-mu)
            cov = cov + alpha_damp*(cov_new-cov)
            
            # update the proposal distribution
            proposal.update_mu(mu)
            proposal.update_cov(cov=cov)

        output_dict = {
            "mu": mu,
            "cov": cov,
            "elbo": elbo_list[-1],
            "mu_traj": mu_params,
            "cov_traj": cov_params,
            "elbo_traj": elbo_list,
            "eps_traj": eps_params,
            "ess_normalized_traj": ess_normalized_list,
            "mean_hat_traj": mean_hat_list,
            "cov_hat_traj": cov_hat_list,
        }
        return output_dict

