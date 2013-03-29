import numpy as np
import math
import scipy
import matplotlib.pyplot as plt

class Chromagram:

    def __init__(self, spectra, sample_rate, bins=12, base=261.63):
        self.sample_rate = sample_rate
        self.bins = bins
        self.base = base
        self.length = spectra.shape[1]
        self.data = np.zeros((self.bins, self.length))

        self._analyse(spectra)

    def _analyse(self, spectra):
        indices = np.mod(
            np.round(
                self.bins * np.log2(
                    (self.sample_rate / 2.0) *
                    np.arange(1, spectra.shape[0]) / 
                    spectra.shape[0] / self.base)
                ), self.bins).astype(int)

        indices = np.insert(indices, 0, 0) # arbitrarily set fq 0 to c

        for i in xrange(self.length):
            self.data[:,i] = np.bincount(
                indices, weights=spectra[:,i])

        self.data = self.data / self.data.max()

    def plot(self):
        fig, ax = plt.subplots()
        self.plot_axes()(ax)
        fig.show()

    def plot_axes(self):
        def function(ax):
            ax.locator_params(axis='y', tight=False, nbins=self.bins + 1)
            ax.imshow(self.data, cmap='gray', interpolation='nearest',
                      origin='lower', aspect='auto')
        return function
