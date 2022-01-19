import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from audio_transport.transport_utils import interpolate, interpolate_old

def plot_spectogram(data, figsize=(8,6)):
    fig, ax = plt.subplots(figsize=figsize)
    S = np.abs(data)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db,y_axis='log', x_axis='time', ax=ax, sr=44100)

    ax.set_title('Power spectrogram')

    fig.colorbar(img, ax=ax, format="%+2.0f dB")


def plot_progression(y1, y2, interp='trivial', mapping=None, figsize=(10,6), size=6):
    """
    Plots the interpolated progression given an interpolation type.
    Interpolates between input signals 
    y1, y2: Input 1d vectors (size n)
    interp: two possible types, 'trivial' and 'displacement'. In the second kind
        a mapping matrix (nxn) is needed 
    size: number of intermediate interpolated signals
    """
    ts = np.linspace(0, 1, size)  
    fig, ax = plt.subplots(1, size, figsize=figsize)
    for t, i in zip(ts, range(0, size)):
        ax[i].plot(y1, ls='--', alpha=0.5, color='k')
        ax[i].plot(y2, ls='--', alpha=0.5, color='k')
        ax[i].axes.yaxis.set_visible(False)
        ax[i].axes.xaxis.set_visible(False)

        if interp == 'trivial':  
            inte = t*y2 + (1-t) * y1
        elif interp == 'displacement':
            assert mapping is not None
            inte = interpolate(mapping, t)
        
        ax[i].plot(inte, label=f't={np.round(t, 1)}',color='tomato')
        ax[i].legend()
