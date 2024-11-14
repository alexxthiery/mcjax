import numpy as np
from .autocorrelation import autocorr


def normalize(samples):
    """return the normalized samples
    obtained by subtracting the mean and
    dividing by the standard deviation"""
    return (samples - np.mean(samples)) / np.std(samples)


def normalized_autocorr(samples):
    """ return the normalized autocorrelation function"""
    if np.std(samples) == 0:
        return np.Infinity

    samples_normalized = normalize(samples)
    rhos = autocorr(samples_normalized)
    return rhos


def IACT_geyer(samples):
    """ Integrated Autocorrelation Time

    IACT = 1 + 2 * sum(autocorr[1:max_lag])

    where max_lag = 2k where (2k) is the smallest k >= 0 such that
       rho[2*k] + rho[2*k+1] is negative.
    """
    rhos = normalized_autocorr(samples)
    sum_autocov = rhos[::2] + rhos[1::2]

    if np.all(sum_autocov >= 0):
        first_time_below_0 = len(rhos)
    else:
        first_time_below_0 = max(np.where(sum_autocov < 0)[0][0] - 1, 0)
    max_lag = 2*first_time_below_0
    IACT = 1 + 2 * np.sum(rhos[1:max_lag])
    return IACT


def ESS_geyer(samples):
    """ Effective Sample Size
    ESS = N / IACT
    """
    assert samples.ndim == 1, f"Error: samples.ndim={samples.ndim}"

    if np.std(samples) == 0:
        return 0.

    N = samples.shape[0]
    IACT = IACT_geyer(samples)
    ESS = N / IACT
    return ESS


def ESS_AR1(
        samples,        # the samples
        threshold=0.5,  # threshold for the autocorrelation computation
        ):
    """ Effective Sample Size
    ESS = N / IACT

    The IACT is computed using the AR(1) approximation.
    The approach consists in finding the smallest k such that
    the autocorrelation at lag k is less than a threshold.

    The autocorrelation is approximated by finding `lambda`
    such that:
        rho[k] = exp(-k / lambda)$
    The IACT is then given by:
        IACT = 1 + 2 sum_{k=1}^{infty} exp(-k/lambda)
             = 1. / tanh(lambda / 2.)
    """
    assert samples.ndim == 1, f"Error: samples.ndim={samples.ndim}"

    if np.std(samples) == 0:
        return 0.

    N = samples.shape[0]
    rhos = normalized_autocorr(samples)

    # find the smallest k such that rho[k] < threshold
    if np.min(rhos) >= threshold:
        ESS = 0.
        return ESS

    max_lag = np.where(rhos < threshold)[0][0]
    lamb = -np.mean(np.log(rhos[1:max_lag]) / np.arange(1, max_lag))
    IACT = 1. / np.tanh(lamb / 2.)
    ESS = N / IACT
    return ESS
