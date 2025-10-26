from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

class BayesianLogisticTarget:
    """
    Target distribution object for Bayesian logistic regression.
    - prior_var: scalar prior variance sigma_w^2
    """
    def __init__(self, prior_var=1.0):
        self.X, self.y = self.load_preprocess_sonar()
        self.prior_var = float(prior_var)
        self.N, self.d = self.X.shape

    @partial(jax.jit, static_argnums=(0,))
    def batch(self, x):
        """
        Evaluate unnormalised log posterior for a batch of weight vectors x.
        x shape: (batch, d)
        returns shape: (batch,)
        log p(x) + sum_i log p(y_i | x)
        """
        logits = self.X @ x.T   
        ll = jnp.sum(self.y[:, None] * logits - jnp.logaddexp(0.0, logits), axis=0) 
        # prior:
        log_prior = -0.5 * jnp.sum(x**2, axis=1) / self.prior_var
        return log_prior + ll

    @partial(jax.jit, static_argnums=(0,))
    def grad_batch(self, x):
        """
        Gradient of log target wrt x. returns shape (batch, d)
        grad_x log p(x|D) = -x/prior_var + sum_i (y_i - sigmoid(u_i^T x)) u_i
        """
        logits = self.X @ x.T  
        probs = jax.nn.sigmoid(logits)   

        coeff = (self.y[:, None] - probs) 
        grad_ll = (coeff.T @ self.X)  
        grad_prior = -x / self.prior_var 
        return grad_prior + grad_ll


    def load_preprocess_sonar(self):
        sonar = fetch_openml(name="sonar", version=1, as_frame=True)
        X = sonar.data.to_numpy()                 
        y = (sonar.target == 'M').astype(np.float32)  # Convert labels 'R'/'M' to 0/1

        # Standardise features and add bias column
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)  

        return jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)