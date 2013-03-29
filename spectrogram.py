import numpy as np
import math
import scipy
import matplotlib.pyplot as plt

class Spectrogram:

    def __init__(self, signal, sample_rate, window_size=1024, 
                 hop_size=512, window_function='hamming'):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_function = window_function
        self.data = np.zeros((self.window_size / 2,
                              math.floor(len(signal) / hop_size)))

        self._analyse(signal)

    def _analyse(self, signal):
        siglen = len(signal)
        siglen = (siglen - self.window_size) - (siglen % self.hop_size)

        for t in xrange(0, siglen, self.hop_size):
            windowed = (signal[t:(t + self.window_size)] *
                        np.hanning(self.window_size))
            spectrum = abs(scipy.fft(windowed))
            spectrum = spectrum[0:len(spectrum) / 2]
            self.data[:, t / self.hop_size] = spectrum

        self.height, self.length = self.data.shape

    def plot(self, log=False, ylim=None):
        fig, ax = plt.subplots()
        self.plot_axes(log, ylim)(ax)
        fig.show()

    def plot_axes(self, log=False, ylim=None):
        if ylim is None:
            ylim = (0, self.sample_rate / 2.0)

        def function(ax):
            nticks = 50
            half_rate = self.sample_rate / 2.0

            if log:
                indices = np.logspace(np.log2(1), np.log2(self.height),
                                      self.height, base=2) - 1
            else:
                indices = np.linspace(0, self.height - 1, self.height)

            freqs = indices * half_rate / self.height
            indices = indices[(freqs >= ylim[0]) & (freqs <= ylim[1])]
            min_ytick = min(indices)
            max_ytick = max(indices)

            if log:
                ytick_labels = np.logspace(np.log2(min_ytick + 1),
                                           np.log2(max_ytick + 1),
                                           nticks, base=2) - 1
            else:
                ytick_labels = np.linspace(min_ytick, max_ytick, nticks)

            data = self.data[np.round(indices).astype(int),]
            ytick_labels *= half_rate / self.height

            ax.imshow(data, cmap='gray', aspect='auto',
                      origin='lower')

            ax.set_yticks(np.linspace(0, len(indices) - 1, nticks))
            ax.set_yticklabels(np.round(ytick_labels).astype(int))
        return function
