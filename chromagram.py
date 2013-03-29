import numpy as np
import math
import scipy
import matplotlib.pyplot as plt

class Chromagram:

    def __init__(self, spectrogram, bins=12, base=261.63):
        self.bins = bins
        self.base = base
        self.length = spectrogram.length
        self.data = np.zeros((self.bins, self.length))

        self._analyse(spectrogram)

    def _analyse(self, spectrogram):
        indices = np.mod(
            np.round(
                self.bins * np.log2(
                    (spectrogram.sample_rate / 2.0) *
                    np.arange(1, spectrogram.height) / 
                    spectrogram.height / self.base)
                ), self.bins).astype(int)

        indices = np.insert(indices, 0, 0) # arbitrarily set fq 0 to c

        for i in xrange(self.length):
            self.data[:,i] = np.bincount(
                indices, weights=spectrogram.data[:,i])

        self.data = self.data / self.data.max()

    def plot(self):
        fig, ax = plt.subplots()
        ax.locator_params(axis='y', tight=False, nbins=self.bins + 1)
        ax.imshow(self.data, cmap='gray', interpolation='nearest',
                  origin='lower', aspect='auto')
        fig.show()
