import numpy as np


def zscore(samples: np.ndarray) -> np.ndarray:
    """
    Standardize the input samples to zero mean and unit variance.

    Parameters:
        samples (np.ndarray): 1D array of samples.

    Returns:
        np.ndarray: Normalized samples.
    """
    mean = np.mean(samples)
    std = np.std(samples)
    if std == 0:
        raise ValueError("Standard deviation is zero. Cannot normalize.")
    return (samples - mean) / std


def autocorr(x: np.ndarray) -> np.ndarray:
    """
    Compute the autocorrelation of a 1D array using FFT for efficiency.

    The output is normalized such that autocorr[0] == 1.

    Parameters:
        x (np.ndarray): Input 1D array.

    Returns:
        np.ndarray: Autocorrelation function, same length as input.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("autocorr expects a 1D array.")

    N = len(x)
    x = x - np.mean(x)  # subtract mean to center the signal

    # Zero-pad to 2N to prevent circular convolution
    fft_x = np.fft.fft(x, n=2*N)
    acf = np.fft.ifft(np.abs(fft_x) ** 2).real[:N]
    acf /= acf[0]  # normalize

    return acf


def normalized_autocorr(samples: np.ndarray) -> np.ndarray:
    """
    Computes the autocorrelation of standardized samples.

    Parameters:
        samples (np.ndarray): 1D array of samples.

    Returns:
        np.ndarray: Autocorrelation function of normalized samples.
    """
    if np.std(samples) == 0:
        raise ValueError("Standard deviation is zero. Autocorrelation undefined.")
    samples_norm = zscore(samples)
    return autocorr(samples_norm)


def iact_geyer(samples):
    """
    Computes the Integrated Autocorrelation Time (IACT) using Geyer's initial positive sequence.

    Parameters:
        samples (np.ndarray): 1D array of samples.

    Returns:
        float: Estimated IACT.
    """
    rhos = normalized_autocorr(samples)
    sum_autocov = rhos[::2] + rhos[1::2]

    if np.all(sum_autocov >= 0):
        first_time_below_0 = len(rhos)
    else:
        first_time_below_0 = max(np.where(sum_autocov < 0)[0][0] - 1, 0)
    max_lag = 2*first_time_below_0
    iact = 1 + 2 * np.sum(rhos[1:max_lag])
    return iact


def ess_geyer(samples):
    """
    Effective Sample Size (ESS) using IACT from Geyer's method.

    Parameters:
        samples (np.ndarray): 1D array of samples.

    Returns:
        float: Effective sample size.
    """
    if samples.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {samples.shape}")
    if np.std(samples) == 0:
        return 0.0
    return samples.shape[0] / iact_geyer(samples)


def ess_ar1(
        samples,        # the samples
        threshold=0.5,  # threshold for the autocorrelation computation
        ):
    """ Effective Sample Size
    ess = N / iact

    The iact is computed using the AR(1) approximation.
    The approach consists in finding the smallest k such that
    the autocorrelation at lag k is less than a threshold.

    The autocorrelation is approximated by finding `lambda`
    such that:
        rho[k] = exp(-k / lambda)$
    The iact is then given by:
        iact = 1 + 2 sum_{k=1}^{infty} exp(-k/lambda)
             = 1. / tanh(lambda / 2.)
    """
    if samples.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {samples.shape}")
    if np.std(samples) == 0:
        return 0.0

    N = samples.shape[0]
    rhos = normalized_autocorr(samples)

    # find the smallest k such that rho[k] < threshold
    if np.min(rhos) >= threshold:
        ess = 0.
        return ess

    max_lag = np.where(rhos < threshold)[0][0]
    lamb = -np.mean(np.log(rhos[1:max_lag]) / np.arange(1, max_lag))
    iact = 1. / np.tanh(lamb / 2.0)
    ess = N / iact
    return ess
