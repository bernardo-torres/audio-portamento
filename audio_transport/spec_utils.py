import librosa
import numpy as np

def stft(wave):
    s = librosa.stft(wave, n_fft=1024, win_length=512, hop_length= int(512 / 2))
    S = np.abs(s)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return s, S_db 
