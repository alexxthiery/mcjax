import jax.scipy.linalg
import jax.numpy as jnp


def is_psd(matrix):
    """
    checks if a matrix is positive semi-definite
    by attempting to compute the cholesky decomposition
    """
    try:
        L = jax.scipy.linalg.cholesky(matrix, lower=True)
        # check that there is any Nan entry
        if jnp.any(jnp.isnan(L)):
            return False
        else:
            return True
    except:
        return False


def log_det_psd(matrix):
    """
    computes the log determinant of a positive semi-definite matrix
    by computing the cholesky decomposition and summing 
    the log of the diagonal elements
    """
    L = jax.scipy.linalg.cholesky(matrix, lower=True)
    log_det = 2 * jnp.sum(jnp.log(jnp.diag(L)))
    return log_det


def invert_psd_matrix(A,            # (dim,dim): psd matrix
                      chol=None,    # (dim,dim): lower triangular Cholesky decomposition of A
                      ):
    """ Invert a positive semi-definite matrix A using the Cholesky decomposition """
    # set the Cholesky decomposition if not provided
    chol = jax.lax.cond(chol is None,
                        lambda _: jnp.linalg.cholesky(A),
                        lambda _: chol if chol is not None else jnp.zeros_like(A),
                        operand=None)

    # compute the inverse of the Cholesky decomposition
    chol_inv = jax.scipy.linalg.solve_triangular(
                                    a=chol,
                                    b=jnp.eye(chol.shape[0]),
                                    lower=True)

    # Compute the inverse of A using L_inv
    A_inv = chol_inv.T @ chol_inv
    return A_inv
