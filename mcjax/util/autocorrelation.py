import numpy as np


def autocorr(x):
    """
    Compute the autocorrelation of the input array using FFT.
    """
    N = len(x)
    # Zero-pad the array to avoid circular convolution
    F = np.fft.fft(x, n=2*N)
    psd = np.abs(F) ** 2
    result = np.fft.ifft(psd)
    result = np.real(result[:N])  # Take the real part and the first N elements
    result /= result[0]  # Normalize the autocorrelation function
    return result