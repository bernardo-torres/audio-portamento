import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from audio_transport.gen_utils import gen_sinewave
from audio_transport.spec_utils import stft
from audio_transport.plot_utils import plot_spectogram, plot_progression
from audio_transport.transport_utils import compute_optimal_map, optimal_1d_mapping, interpolate, join_stfts

x1 = gen_sinewave(440, 4, 44100) * 0.2 
x2 = gen_sinewave(660, 4, 44100) * 0.2

s1, S_db1 = stft(x1)
s2, S_db2 = stft(x2)

new_D = join_stfts(s1[:, :140], s2[:, 20:], 100)
plot_spectogram(new_D, figsize=(20,15))