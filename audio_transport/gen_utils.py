import numpy as np

def gen_sinewave(freqs, total_time, sample_rate):
    """
    Generates sine waves
    """
    if type(freqs) == list:
        time_values = np.arange(0, total_time, 1 / sample_rate )
        signal = np.sin(2 *np.pi * freqs[0] * time_values)
        for f in freqs[1:]:
            signal = signal + np.sin(2 *np.pi * f * time_values)
        return signal
    elif type(freqs) == int:
        time_values = np.arange(0, total_time, 1 / sample_rate )
        return np.sin(2 *np.pi * freqs * time_values)
    else:
        print('unsuported type')
        return []


def gen_gaussian(mu, sigma, N):
    """
    Generates gaussian on interval [0,1],
    centered on mu with std of sigma,
    composed of N points
    """
    t = np.arange(0,N)/N
    gauss = np.exp(-(t-mu)**2/(2*sigma**2))
    return gauss

def normalize(x, vmin=0.2):
    x = x + np.max(x) * vmin
    return x/np.sum(x)