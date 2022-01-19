import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import soundfile as sf
import librosa

import audio_transport.plot_utils as plot_utils
from audio_transport.gen_utils import normalize

WIN_LENGTH = 2206
HOP = int(WIN_LENGTH / 2)
N_FFT = 4096
SR = 44100

def stft(wave):
    s = librosa.stft(wave, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length= HOP)
    S = np.abs(s)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return s, S_db 


def optimal_1d_mapping(alpha, beta):
    """
    Computes optimal matching problem solution given two 1d input vectors
    """
    # Sorts alpha and beta
    perm_a = np.argsort(alpha)
    perm_b = np.argsort(beta)

    map1 = alpha[perm_a]

    inv_p = np.zeros(len(perm_a))
    inv_p[perm_a] = np.arange(0, len(perm_a))

    sigma = perm_b[inv_p.astype(int)]
    return sigma

def distmat(x,y):
    #return np.linalg.norm(x - y) ** 2
    #return np.sum(x**2,0)[:,None] + np.sum(y**2,0)[None,:] - 2*x.transpose().dot(y)
    C = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            C[i, j] = (i - j) ** 2
    return C

def compute_optimal_map_lp(a, b, n, m):
    """
    Computes the optimal mapping using a linear program solver (cvxpy)
    """
    import cvxpy as cp

    C = distmat(a,b)
    P = cp.Variable((n,m))
    u = np.ones((m,1))
    v = np.ones((n,1))
    U = [0 <= P, cp.matmul(P,u)==a.reshape(n,1), cp.matmul(P.T,v)==b.reshape(n,1)]

    objective = cp.Minimize( cp.sum(cp.multiply(P,C)) )
    prob = cp.Problem(objective, U)
    result = prob.solve(verbose=False)

    return P.value

def compute_optimal_map(x, y):
    """
    Computes 1D optimal transport map using north-west corner rule heuristic
    Runs in O(2*|n|) time
    """
    n = len(x)
    if len(x) != len(y):
        print('dimensions are not equal')
        return []
    
    pi = np.zeros((n, n))

    # Compute initial mass containers
    px = np.abs(x[0])
    py = np.abs(y[0])
    (i, j) = (0, 0)
    while True:
        # If there is less mass in container px, 
        # transfer as much as possible from py
        if (i >= n) and (j >= n):
            break
        if px < py:
            if (i < n ):
                pi[i, j] = px
                py = py - px
                i = i + 1
                if i >= n:
                    break
                # Refills x container with next mass
                px = np.abs(x[i])
                
        else:
            if (j < n ):
                pi[i, j] = py
                px = px - py
                j = j + 1
                if j >= n:
                    break
                # Refills x container with next mass
                py = np.abs(y[j])
                
    return pi

def interpolate(p, t, mass1=None, mass2=None):
    """
    Computes spectral interpolation between x and y

    Inputs:
    p: optimal mapping computed from normalized 1d signals (n,n)
    t: interpolation factor in range [0,1] 
    mass1: total mass of input base measure (if None 
        interpolation will not take mass into account)
    mass2: total mass of input target measure

    Returns:
    interp: interpolated measure (1D vector, size n)
    """

    n = p.shape[0]
    interp = np.zeros(n)

    # Finding non-zero entries of the map
    I,J = np.nonzero(p>1e-5)

    # Computes the displaced frequency 
    # Also rounds it to the newest integer
    k = (1 - t) * I +  t *(J) 
    k_floor = np.floor(k).astype(int)  # Round down
    k_ceil = np.ceil(k).astype(int)  # Round up
    v = k - k_floor  # diff between displaced frequency and closest lower original frequency

    # Iterates over non-zero entries of the transport map
    for (i, j, l) in zip(I, J, np.arange(0, len(I))):    

        if (k_ceil[l] < n-1):
            # Transfers mass proportionally to nearest frequencies 
            # to the right and left of displaced frequency
            interp[k_floor[l]] = interp[k_floor[l]] + p[i, j] * (1 - v[l])
            interp[k_ceil[l]] = interp[k_ceil[l]] + p[i, j] * v[l]

        elif (k_floor[l] == 0) or (k_ceil[l] == n-1):
            interp[k_floor[l]] = interp[k_floor[l]] + p[i, j] 
        elif k_ceil[l] == n:
            interp[k_ceil[l] - 1] = interp[k_ceil[l] - 1] + p[i, j] 
    
    # Uses mass information to get proportional mapping
    if mass1 != None:
        interp = interp * mass1 * (1 - t) + interp * mass2 * (t)
    return interp



def join_stfts(s1, s2, n_windows, verbose=True, correct_phase='repeat'):
    """
    Joins stfts s1 and s2 while interpolating in the middle 

    Inputs:
    s1: complex spectogram of input audio 1 (size (D, n1))
    s1: complex spectogram of input audio 2 (size (D, n2))
    n_windows: number of frames to interpolate

    Returns:
    new_spec: complex spectogram (size D, (n1+n_windows+n2))
    """

    if (s1.shape[0] != s2.shape[0]):
        print("Different number of frequency bins")

    new_spec = np.empty((s1.shape[0], s1.shape[1] + s2.shape[1] + n_windows), dtype=complex)

    # Fills in spectrum from first clip
    new_spec[:, :s1.shape[1]] = s1[:, :]

    # Fills in spectrum from second clip
    new_spec[:, s1.shape[1] + n_windows :] = s2[:, :]

    alpha = s1[:, -1]  # spectrum of last frame of s1
    beta = s2[:, 0]  # spectrum of first frame of s2

    # Total spectral mass of both spectra
    mass = lambda x: np.sum(np.abs(x))
    mass1 = mass(alpha)
    mass2 = mass(beta)
    norm_alpha = alpha / mass1
    norm_beta = beta / mass2

    # Optimal mapping of normalized spectra
    p = compute_optimal_map(norm_alpha, norm_beta)

    if verbose:
        print("Mass of signal 1: " + str(mass1))
        print("Mass of signal 2: " + str(mass2))


    # Computes displacement interpolation
    ts = np.linspace(0, 1, n_windows)
    phi_prev = np.angle(alpha)
    for t, i in zip(ts, range(0, n_windows)):

        # Interpolated spectrum magnitude
        interp_abs = np.abs(interpolate(p, t, mass1=mass1, mass2=mass2))
        #np.angle(new_spec[:, s1.shape[1] + i - 1]) + 
        if correct_phase == 'repeat':
            interp_phase = phi_prev
        elif correct_phase == 'zero' or correct_phase == None:
            interp_phase = 0
        elif correct_phase == 'vocoder':
            # Phase computation
            freqs = SR / N_FFT / 2 * np.arange(0, len(alpha))
            #print(phi_prev)
            #phi = 2 * np.pi * freqs * WIN_LENGTH / SR  +  phi_prev
            phi = phi_prev % 2 * np.pi
            phi_prev = phi
            interp_phase = interp_abs * np.sin(phi)
       
        # Fills in interpolated spectrum
        new_spec[:, s1.shape[1] + i] = interp_abs + interp_phase * 1j


    #print(np.angle(new_spec[:, s1.shape[1] -1][:5]))
    #print(s1.shape)
    #print(np.angle(new_spec[:, s1.shape[1]][:5]))
    #plt.plot( np.abs(alpha[:50]))
    #plt.plot( np.abs(norm_alpha[:50]) *mass1*2)
    return new_spec



def transport(x1, x2, t1, t2, t3, sr=44100, size_window=2206, correct_phase='repeat', plot=None, write_file=None):
    """
    Computes spectral interpolation between two audio signals using 1D optimal transport
    Spectral interpolation is computed using displacement interpolation.

    Inputs:
    x1: raw audio signal 1 
    x2: raw audio signal 1
    t1: number of seconds x1 will play
    t2: number of seconds x1 will play
    t3: number of seconds used to do interpolation
    sr: sampling rate (Hz)
    size_window: FFT window (frame) length
    write_file: complete name of resulting audio file (ex. text.wav)
    """
    # Number of frames needed to play requested times
    size_window = int(size_window / 2)
    n_windows1 = int(t1 * sr / size_window)
    n_windows2 = int(t2 * sr / size_window)
    n_windows3 = int(t3 * sr / size_window)

    # Compute stfts of input signals
    s1, S_db1 = stft(x1)
    s2, S_db2 = stft(x2)

    # Exclude some frames to avoid boundary effects
    begin = 10
    end = 10

    new_D = join_stfts(s1[:, :n_windows1 - end], s2[:, begin:n_windows2], n_windows3, correct_phase=correct_phase)
    
    if plot==1:
        plot_utils.plot_spectogram(s1, figsize=(12,8))
    elif plot==2:
        plot_utils.plot_spectogram(s2, figsize=(12,8))
    elif plot==3:
        plot_utils.plot_spectogram(new_D, figsize=(12,8))

    # Computes ifft
    I = librosa.istft(new_D, win_length=WIN_LENGTH, hop_length= HOP)

    if write_file != None:
        sf.write(write_file, I, SR)
    return

# Obsolete
def interpolate_old(x, y, p, t):

    n = len(x)
    interp = np.zeros(n)
    for i in range(0, n):
        for j in range(0, n):
            k = int((1 - t) * i +  t *(j))
            interp[k] = interp[k] + p[i, j]
        #print(i)
    return interp