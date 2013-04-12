import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
import scipy.misc
import scipy.weave

class Spectrogram(object):

    def __init__(self, signal, sample_rate, window_size=4096, 
                 hop_size=2048, window_function='hamming'):
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

        window = np.hanning(self.window_size)
        for t in xrange(0, siglen, self.hop_size):
            windowed = signal[t:(t + self.window_size)] * window
            spectrum = abs(scipy.fft(windowed))
            spectrum = spectrum[0:len(spectrum) / 2]
            self.data[:, t / self.hop_size] = spectrum

        self.height, self.length = self.data.shape

    def plot(self, *args, **kwargs):
        fig, ax = plt.subplots()
        self.plot_axes(*args, **kwargs)(ax)
        fig.show()

    def plot_axes(self, log=False, ylim=None, max_dim=(400, 400)):
        if ylim is None:
            ylim = (0, self.sample_rate / 2.0)

        def function(ax):
            nticks = 50
            half_rate = self.sample_rate / 2.0

            dim = map(min, zip(self.data.shape, max_dim))
            data = scipy.misc.imresize(self.data, dim)

            height, length = dim

            if log:
                indices = np.logspace(np.log2(1), np.log2(height),
                                      height, base=2) - 1
            else:
                indices = np.linspace(0, height - 1, height)

            freqs = indices * half_rate / height
            indices = indices[(freqs >= ylim[0]) & (freqs <= ylim[1])]
            min_ytick = min(indices)
            max_ytick = max(indices)

            if log:
                ytick_labels = np.logspace(np.log2(min_ytick + 1),
                                           np.log2(max_ytick + 1),
                                           nticks, base=2) - 1
            else:
                ytick_labels = np.linspace(min_ytick, max_ytick, nticks)

            data = data[np.round(indices).astype(int),]
            ytick_labels *= half_rate / height

            ax.imshow(data, cmap='binary', aspect='auto',
                      origin='lower')

            ax.set_yticks(np.linspace(0, len(indices) - 1, nticks))
            ax.set_yticklabels(np.round(ytick_labels).astype(int))

        return function


def monophonic_path(spectrogram_data):
    values = np.max(spectrogram_data) - spectrogram_data
    costs = np.zeros(spectrogram_data.shape)
    prev = np.zeros(spectrogram_data.shape)
    amp = amplitude(spectrogram_data)

    change_cost = .1
    silence_cost = 2

    costs[:, 0] = values[:, 0]

    height, length = values.shape

    with open('dynprog.c', 'r') as f:
        code = f.read()
    scipy.weave.inline(code, ['values', 'costs', 'prev', 'change_cost',
                              'amp', 'silence_cost', 'length', 'height'],
                       type_converters=scipy.weave.converters.blitz)

    path = [0] * values.shape[1]
    path[-1] = np.argmin(costs[:, -1])
    for t in reversed(xrange(len(path) - 1)):
        path[t] = prev[path[t + 1], t]

    return path

def monophonic_maxima_path(spectrogram_data):
    return spectrogram_data.argmax(0)

# notes is a matrix with fields [start, end, note]
def notes_from_path(path, threshold=1):
    notes = np.empty((0, 3))
    current = []
    start_time = 0
    for t, x in enumerate(path):
        if len(current):
            mean = sum(current) / float(len(current))
        else:
            mean = 0

        if abs(x - mean) > threshold:
            if mean > 0:
                notes = np.vstack((notes, [start_time, t, round(mean)]))
            current = []
            start_time = t + 1

        if x > 0:
            current.append(x)

    if mean > 0:
        notes = np.vstack((notes, [start_time, t, round(mean)]))

    notes = notes.astype(int)
    return notes

def notes_quantise_pitch_class(note_bins, octave_steps, sample_rate, window_size):
    notes = note_bins.astype(float)
    freqs = note_bins * sample_rate / window_size
    base = 261.63 # c, TODO: should probably be something less western
    pitches = np.log2(freqs / base) * octave_steps
    classes = np.mod(pitches, octave_steps)
    classes = np.round(classes).astype(int)

    return classes


# assuming argmax ioi = beat
# returns matrix with columns [time, klang_1, klang_2, [...], klang_n]
def get_nklangs(notes, nbeats, n, threshold=.5):
    iois = notes[1:, 0] - notes[:-1, 0]
    beat = np.argmax(np.bincount(iois))
    bars = []
    bar = []
    phase = 0
    time = 0

    for i, ioi in enumerate(iois):
        if ioi > beat * threshold:
            bar.append((ioi, notes[i, 2]))

        if phase > beat * nbeats:
            if len(bar) > 0:
                bars.append((time, bar))
            bar = []
            phase -= beat * nbeats

        phase += ioi
        time += ioi

    nklangs = np.zeros((len(bars), n + 1))
    for i, (time, bar) in enumerate(bars):
        nklang = [x[1] for x in sorted(bar, reverse=True)[0:n]]
        if len(nklang) < n:
            nklang = [nklang[0]] * (n - len(nklang) + 1) + nklang[1:]
        nklangs[i,:] = [time] + nklang

    nklangs = nklangs.astype(int)

    return nklangs

def nklangs_to_feature_vector(nklangs, octave_steps):
    n = nklangs.shape[1] - 1
    values = (nklangs[:, 1:] * np.power(octave_steps, np.arange(n))).sum(1)
    vector = np.bincount(values.astype(int))
    #length = np.sum(([octave_steps] * n) * np.power(octave_steps, np.arange(n)))
    # closed form
    length = (octave_steps * (octave_steps ** n - 1) / (octave_steps - 1))
    return pad1d(vector, 0, length)

def amplitude(spectrogram_data, smooth_width=50):
    window = np.hanning(smooth_width)
    window /= window.sum()

    amp = np.sum(spectrogram_data[1:,:], 0) # ignore 0 index
    return np.convolve(amp, window, mode='same')

def pad1d(array, x, length):
    return np.append(array, [x] * (length - len(array)))

def audio_to_feature_vector(filename):
    import audio
    a = audio.Audio(filename)
    s = Spectrogram(a.signal[:, 0], a.sample_rate, 4096, 2048)
    path = monophonic_maxima_path(s.data)
    notes = notes_from_path(path)
    steps = 12
    classes = notes_quantise_pitch_class(notes[:, 2], steps, s.sample_rate, s.window_size)
    notes[:,2] = classes
    nklangs = get_nklangs(notes, 4, 2)
    fv = nklangs_to_feature_vector(nklangs, steps)
    return nklangs, fv
