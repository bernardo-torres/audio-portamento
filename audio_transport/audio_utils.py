import numpy as np
from scipy import signal
from math import ceil

def ChangeTimeScale(alpha, A, Fs):
    """
    Inputs:
        alpha: multiplication factor
        A: analysis marks
        Fs: sampling frequency
    Returns:
        B: synthesis marks
    """
    B = np.zeros((2, int(A.shape[1] * alpha)))
    ts = 0
    nk = 0
    
    for k in np.arange(len(B[0])):
        pa = A[2][int(np.floor(nk))]
        ts = ts + pa  # Computes synthesis mark
        B[0][k] = ts
        B[1][k] = nk
        nk = nk + 1 / alpha
    return B

def nextpow2(x):
    assert x>0
    p = ceil(np.log2(x))
    x_ = 2**p
    assert 2**(p-1) < x <= x_
    return x_
    
def period(x, Fs, Pmin=1/300, Pmax=1/80, seuil=0.7) :
    # [P,voiced] = period(x,Fs,Pmin,Pmax,seuil);
    # If voiced = 1, P is the period signal x expressed in number of samples
    # If voiced = 0, P is equal to 10ms.Fs

    x = x - np.mean(x)
    N = len(x)

    Nmin = np.ceil(Pmin*Fs).astype(int)
    Nmax = 1 + np.floor(Pmax*Fs).astype(int)
    Nmax = np.min([Nmax,N])

    Nfft = nextpow2(2*N-1)
    X = np.fft.fft(x, n=Nfft)
    S = X * np.conj(X) / N
    r = np.real(np.fft.ifft(S))

    rmax = np.max(r[Nmin:Nmax])
    I = np.argmax(r[Nmin:Nmax])
    P = I+Nmin
    corr = (rmax/r[0]) * (N/(N-P))
    voiced = corr > seuil
    if not(voiced):
        P = np.round(10e-3*Fs)

    return P,voiced

def AnalysisPitchMarks(s, Fs):
    """
    Inputs:
        s: data vector 
        Fs: sampling frequency
    Returns
        result: Array, 
            first row = times corresponding to analysis marks
            second row = boolean indicating if signal is voiced in 
                the neighborhood of analysis mark
            third row: pitch corresponding to the mark 
                (period expressed in number of samples, 10ms * Fs if unvoiced)
    """
    # Init values t0 = 0 and p0 = Fs * 10 ms
    tn = 0
    pn = Fs / 100
    end = tn + int(2.5 * pn)

    voiced_results = []
    times_results = []
    pitch_results = []
    while end < len(s) - 1:
        # End position of sequence of duration 2.5 Pa(n-1)
        end = tn + int(2.5 * pn)
        if (end >= len(s)):
            end = len(s) - 1

        # Extracting sequence
        x = s[tn:end]

        # Computes estimated pitch (period) for sequence
        pn, voiced = period(x, Fs)

        # Sets next analysis mark
        tn = int(tn + pn)
        
        voiced_results.append(voiced)
        times_results.append(tn)
        pitch_results.append(pn)
    result = np.array([times_results, voiced_results, pitch_results])
    return result

def Synthesis(s, Fs, A, B):
    """
    Inputs:
        s: original signal
        Fs: sampling frequency
        A: Matrix with first row containing the analysis marks
        B: Matrix with first row containing the synthesis marks
    Returns:
        y: synthesized signal
    """
    
    # Initializes y of size ts(kend) + Pa(n(kend))
    nkend = int(B[1][-1])
    y = np.zeros(int(B[0][-1] + A[2][nkend]))
    
    for k in np.arange(0, len(B[0])):
        nk = int(B[1][k])
        pa = A[2][nk]  # pa(n(k)) -> n(k) being index of analysis mark
        ta = A[0][nk]  # ta(n(k)) -> time corresponding to analysis mark n
        ts = B[0][k]  # ts(k) -> time corresponding to synthesis mark k
        
        # Extracts window of length 2pa + 1 centered on analysis mark ta(n(k))
        beg = int(ta - pa)
        end = int(ta + pa)
        x = s[beg:end]
        
        # Multiplies by hann window
        x = x * signal.hann(len(x))
        
        # Overlap add of sequence on output vector around synthesis mark ts
        y[int(ts - pa):int(ts + pa)] = y[int(ts - pa):int(ts + pa)] + x
        
    return y

