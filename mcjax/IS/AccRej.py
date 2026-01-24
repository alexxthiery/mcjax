import jax
import jax.numpy as jnp
import jax.random as jr
from mcjax.proba.density import LogDensity


""" Accept/Reject Sampling """
class AccRej:
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
            n_iter: int,                 # number of iterations
            proposal: LogDensity,           # proposal distribution
            m: int,                         # factor M
            verbose: bool = False
            ):
        
        # to store all the samples
        samples = []

        # accepted samples
        acc_samples = []
        
        # to store the log_target values
        log_target_values = []

        # to store the accepted log_target values
        log_target_values_acc = []

        # to store the log_proposal values
        log_proposal_values = []

        # to store the accepted log_proposal values
        log_proposal_values_acc = []
        
        # save the list of accept/reject
        acc_list = []

        for it in range(n_iter):
            if verbose:
                print(f"Iteration {it+1}/{n_iter}")
            
            # sample from proposal density
            key, key_ = jr.split(key)
            x = proposal.sample(key=key_, n_samples=1)

            error_msg = "Target density exceeds m*proposal density"
            prop_density = proposal.logdensity(x)
            target_density = self.logtarget.logdensity(x)
            assert jnp.log(m) + prop_density >= target_density, error_msg

            samples.append(x)
            log_target_values.append(target_density)
            log_proposal_values.append(prop_density)
            
            # accept or reject
            key, key_ = jr.split(key)
            u = jr.uniform(key_)
            acc = u < jnp.exp(target_density - prop_density - jnp.log(m))
            if acc:
                acc_samples.append(x)
                log_target_values_acc.append(target_density)
                log_proposal_values_acc.append(prop_density)

            acc_list.append(acc)
        
        dist_output = {
            "samples": samples,
            "acc_samples": acc_samples,
            "log_target_values": log_target_values,
            "log_target_values_acc": log_target_values_acc,
            "log_proposal_values":log_proposal_values,
            "log_proposal_values_acc":log_proposal_values_acc,
            "acc_list":acc_list
        }
        return dist_output


            