import os
import mad
import numpy as np
import struct
import scipy.io.wavfile
import operator
import math

from util import *

class Audio(object):

    def __init__(self, path=None, max_time=None, sample_rate=None, channels=None):

        self.signal = None
        self.remote_path = None

        if sample_rate:
            self.sample_rate = sample_rate

        if channels:
            self.channels = channels

        if not path:
            return

        self.max_time = max_time
        
        if path.startswith('http://'):
            self.remote_path = path
            self.filename = download(path)
        else:
            self.remote_path = None
            self.filename = path

        ext = get_extension(path)
        if ext == '.mp3':
            self._read_mp3()
        elif ext == '.wav':
            self._read_wav()
        else:
            raise NotImplementedError('Unknown file extension: %s (%s)' %
                                      (ext, path))

    def __del__(self):
        if self.remote_path:
            os.unlink(self.filename)

    def _read_mp3(self):
        mf = mad.MadFile(self.filename)

        if mf.mode() == mad.MODE_SINGLE_CHANNEL:
            self.channels = 1
        elif (mf.mode() == mad.MODE_JOINT_STEREO or
              mf.mode() == mad.MODE_STEREO):
            self.channels = 2
        else:
            raise NotImplementedError('Unsupported stereo mode')

        self.sample_rate = mf.samplerate()
        if self.max_time is not None:
            max_samples = self.sample_rate * self.max_time
        else:
            max_samples = None

        signal_l = []
        if self.channels == 2:
            signal_r = []

        length = 0

        while True:
            buf = mf.read()
            if buf is None:
                break

            nsamples = len(buf) / 2 # 2 chars per short
            #nsamples = len(buf) / 4 # 4 chars per int
            length += nsamples / self.channels

            buf = struct.unpack_from('%dh' % nsamples, buf)
            buf = [x / 32768.0 for x in buf]
            #buf = struct.unpack_from('%di' % nsamples, buf)
            #buf = [x / 4294967296.0 for x in buf]

            if self.channels == 1:
                signal_l.extend(buf)
            elif self.channels == 2:
                signal_l.extend(buf[0::2])
                signal_r.extend(buf[1::2])

            if max_samples is not None and length >= max_samples:
                break

        if self.channels == 1:
            self.signal = np.array([signal_l], dtype='float32')
        elif self.channels == 2:
            self.signal = np.array([signal_l, signal_r], dtype='float32')

        if max_samples is not None and self.signal.shape[0] > max_samples:
            self.signal = self.signal[0:max_samples, :]

        self.signal = np.transpose(self.signal)

    def _read_wav(self):
        self.sample_rate, self.signal = scipy.io.wavfile.read(self.filename)
        if self.signal.dtype == 'int16':
            self.signal = self.signal.astype(float) / 32768.0
        elif self.signal.dtype == 'int32':
            self.signal = self.signal.astype(float) / 4294967296.0

        if max(self.signal) > 1 or min(self.signal) < -1:
            raise Exception('Unknown data type')

        sig_shape = self.signal.shape
        if len(sig_shape) == 1:
            sig_shape = (sig_shape[0], 1)
            self.signal.shape = sig_shape

        self.channels = self.signal.shape[1]

    def length(self):
        if self.signal is not None:
            return self.signal.shape[0]
        return 0
            
    def play(self):
        import alsaaudio
        pcm = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK)
        pcm.setchannels(self.channels)
        pcm.setformat(alsaaudio.PCM_FORMAT_FLOAT_LE)
        pcm.setrate(self.sample_rate)

        period_size = 128
        pcm.setperiodsize(period_size)

        for pos in xrange(0, self.length(), period_size):

            period = min(period_size, self.length() - pos)
            end_pos = pos + period

            if self.channels == 1:
                signal = self.signal[pos:end_pos, 0]
            elif self.channels == 2:
                signal = np.empty((period * 2,), dtype=self.signal.dtype)
                signal[0::2] = self.signal[pos:end_pos, 0]
                signal[1::2] = self.signal[pos:end_pos, 1]

            #signal = [int(x * 32768) for x in signal]

            signal = struct.pack('<%df' % len(signal), *signal)

            pcm.write(signal)

    def plot(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(self.channels, 1, sharex=True, sharey=True)
        if self.channels == 1:
            ax = (ax,)
        [self.plot_axes(i)(ax[i]) for i in range(self.channels)]
        fig.show()

    def plot_axes(self, channel=0):
        def function(ax):
            ax.plot(self.signal[:, channel])
        return function

    def add_note(self, start, duration, frequency, amplitude=1.0, waveform='sine'):
        if waveform != 'sine':
            raise Exception('Unknown waveform: %s' % waveform)

        end = start + duration

        if self.signal is None:
            self.signal = np.zeros((end, self.channels))
        elif self.signal.shape[0] < end:
            self.signal = np.vstack((self.signal, np.zeros((
                            end - self.signal.shape[0], self.channels))))

        sig = np.sin(np.arange(duration) * 2 * math.pi *
                     frequency / self.sample_rate) * amplitude

        for i in range(self.channels):
            self.signal[start:end,i] += sig

    def clone(self):
        new = Audio()
        new.sample_rate = self.sample_rate
        new.channels = self.channels
        new.signal = self.signal.copy()
        return new

def merge(first, second):
    first = first.clone()
    if first.sample_rate != second.sample_rate:
        raise Exception('Cannot merge Audio objects with different sample rates')

    if first.channels < second.channels:
        first.channels = second.channels
        for i in range(second.channels - first.channels):
            first.signal = np.hstack((first.signal, first.signal[:,0]))

    if first.length() < second.length():
        padding = np.zeros((second.length() - first.length(),
                            first.channels))
        first.signal = np.vstack((first.signal, padding))

    for i in range(first.channels):
        first.signal[:second.length(), i] += second.signal[:, i % second.channels]
    return first
