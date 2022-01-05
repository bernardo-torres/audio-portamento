import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import soundfile as sf
import librosa

from audio_transport.plot_utils import plot_spectogram
from audio_transport.gen_utils import normalize


def stft(wave):
    s = librosa.stft(wave, n_fft=1024, win_length=512, hop_length= int(512 / 2))
    S = np.abs(s)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return s, S_db 


def optimal_1d_mapping(alpha, beta):
    """
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
    C = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            C[i, j] = (i - j) ** 2
    return C

def compute_optimal_map_lp(a, b, n, m):
    import cvxpy as cp
    
    C = distmat(a,b)
    P = cp.Variable((n,m))
    u = np.ones((m,1))
    v = np.ones((n,1))
    U = [0 <= P, cp.matmul(P,u)==a.reshape(n, 1), cp.matmul(P.T,v)==b.reshape(m, 1)]

    objective = cp.Minimize( cp.sum(cp.multiply(P,C)) )
    prob = cp.Problem(objective, U)
    result = prob.solve()

    return P.value

def compute_optimal_map(x, y):
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

def interpolate(x, y, p, t, mass1=None, mass2=None):

    n = len(x)
    interp = np.zeros(n)
    # Finding non-zero entries of the map
    I,J = np.nonzero(p>1e-5)

    # Computes the displaced frequency 
    # Also rounds it to the newest integer
    k = (1 - t) * I +  t *(J) 
    k_floor = np.floor(k).astype(int)
    k_ceil = np.ceil(k).astype(int)
    v = k - k_floor
    for (i, j, l) in zip(I, J, np.arange(0, len(I))):
        
        # Finds new t given the rounded interpolated bin
        if (k_ceil[l] < n-1):
            #print(k_ceil[l])
            interp[k_floor[l]] = interp[k_floor[l]] + p[i, j] * (1 - v[l])
            interp[k_ceil[l]] = interp[k_ceil[l]] + p[i, j] * v[l]
        elif (k_floor[l] == 0) or (k_ceil[l] == n-1):
            interp[k_floor[l]] = interp[k_floor[l]] + p[i, j] 
        elif k_ceil[l] == n:
            interp[k_ceil[l] - 1] = interp[k_ceil[l] - 1] + p[i, j] 
    
    if mass1 != None:
        interp = interp * mass1 * (1 - t) + interp * mass2 * (t)
    return interp



def join_stfts(s1, s2, n_windows):
    """
    """
    if (s1.shape[0] != s2.shape[0]):
        print("Different number of frequency bins")
    new_spec = np.empty((s1.shape[0], s1.shape[1] + s2.shape[1] + n_windows), dtype=complex)
    print(new_spec.shape, s1.shape, s2.shape)
    # Fills in spectrum from first clip
    new_spec[:, :s1.shape[1]] = s1[:, :]

    # Fills in spectrum from second clip
    new_spec[:, s1.shape[1] + n_windows :] = s2[:, :]

    alpha = s1[:, -1]
    beta = s2[:, 0]

    # Fills in interpolated spectrum
    ts = np.linspace(0, 1, n_windows)
 
    #sigma = optimal_1d_mapping(alpha, beta)
    #beta2 = beta[sigma]

    mass = lambda x: np.sum(np.abs(x))
    mass1 = mass(alpha)
    mass2 = mass(beta)
    norm_alpha = normalize(alpha)
    norm_beta = normalize(beta)
    p = compute_optimal_map(norm_alpha, norm_beta)
    

    #new_spec[:, s1.shape[1]] = new_spec[:, s1.shape[1]-1] / mass1
    #new_spec[:, s1.shape[1] + n_windows] = new_spec[:, s1.shape[1] + n_windows] / mass2
    print(mass1, mass(normalize(alpha)))
    print(mass2, mass(normalize(beta)))

    for t, i in zip(ts, range(0, n_windows)):
        #new_spec[:, s1.shape[1] + i] = ((1-t) * alpha + t * beta ) 
        new_spec[:, s1.shape[1] + i] = interpolate(norm_alpha, norm_beta, p, t, 
                            mass1=mass1, mass2=mass2)

    plt.plot( alpha[:50])
    plt.plot( beta[:50])
    # int(n_windows/2)
    plt.plot(new_spec[:50, s1.shape[1] ])
    #print(mass1, mass(normalize(alpha)))
    #print(mass2, mass(normalize(beta)))
    return new_spec



def transport(x1, x2, t1, t2, t3, sr, size_window=512, plot=None, write_file=None):

    n_windows1 = int(t1 * sr / size_window)
    n_windows2 = int(t2 * sr / size_window)
    n_windows3 = int(t3 * sr / size_window)

    # Compute stfts of input signals
    s1, S_db1 = stft(x1)
    s2, S_db2 = stft(x2)


    # Exclude some frames to avoid boundary effects
    begin = 50
    end = 50

    # 
    new_D = join_stfts(s1[:, :n_windows1 - end], s2[:, begin:n_windows2], n_windows3)

    
    if plot==1:
        plot_spectogram(s1, figsize=(12,8))
    elif plot==2:
        plot_spectogram(s2, figsize=(12,8))
    elif plot==3:
        plot_spectogram(new_D, figsize=(12,8))

    # Computes ifft
    I = librosa.istft(new_D, win_length=512, hop_length= int(512 / 2))

    if write_file != None:
        sf.write(write_file, I, 44100)

    return