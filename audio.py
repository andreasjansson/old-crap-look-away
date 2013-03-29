import os
import mad
import numpy as np
import struct
import scipy.io.wavfile
import matplotlib.pyplot as plt

from util import *

class Audio(object):

    def __init__(self, path=None):
        if not path:
            return
        
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
            raise NotImplementedError('Unknown file extension')

    def __del__(self):
        if self.remote_path:
            os.unlink(self.filename)

    def _read_mp3(self):
        mf = mad.MadFile(self.filename)

        if mf.mode() == mad.MODE_SINGLE_CHANNEL:
            self.channels = 1
        elif mf.mode() == mad.MODE_JOINT_STEREO:
            self.channels = 2
        else:
            raise NotImplementedError('Unsupported stereo mode')

        self.sample_rate = mf.samplerate()

        signal_l = []
        if self.channels == 2:
            signal_r = []

        length = 0
        while True:
            buf = mf.read()
            if buf is None:
                break

            nsamples = len(buf) / 2 # 2 chars per short
            length += nsamples / self.channels

            buf = struct.unpack_from('%dh' % nsamples, buf)
            buf = [x / 32768.0 for x in buf]

            if self.channels == 1:
                signal_l.extend(buf)
            elif self.channels == 2:
                signal_l.extend(buf[0::2])
                signal_r.extend(buf[1::2])

        self.length = length
        if self.channels == 1:
            self.signal = np.array([signal_l])
        elif self.channels == 2:
            self.signal = np.transpose(np.array([signal_l, signal_r]))

    def _read_wav(self):
        self.sample_rate, self.signal = scipy.io.wavfile.read(self.filename)
        if self.signal.dtype == 'int16':
            self.signal = self.signal.astype(float) / 32768.0

        if max(self.signal) > 1 or min(self.signal) < -1:
            raise Exception('Unknown data type')

        sig_shape = self.signal.shape
        if len(sig_shape) == 1:
            sig_shape = (sig_shape[0], 1)
            self.signal.shape = sig_shape

        self.channels = self.signal.shape[1]
        self.length = self.signal.shape[0]
            
    def play(self):
        import alsaaudio
        pcm = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK)
        pcm.setchannels(self.channels)
        pcm.setformat(alsaaudio.PCM_FORMAT_FLOAT_LE)
        pcm.setrate(self.sample_rate)

        period_size = 128
        pcm.setperiodsize(period_size)
        
        for pos in xrange(0, self.length, period_size):

            period = min(period_size, self.length - pos)
            end_pos = pos + period

            if self.channels == 1:
                signal = self.signal[pos:end_pos]
            elif self.channels == 2:
                signal = np.empty((period * 2,), dtype=self.signal.dtype)
                signal[0::2] = self.signal[pos:end_pos, 0]
                signal[1::2] = self.signal[pos:end_pos, 1]

            #signal = [int(x * 32768) for x in signal]
            signal = struct.pack('<%df' % len(signal), *signal)

            pcm.write(signal)

    def plot(self):
        fig, ax = plt.subplots(self.channels, 1, sharex=True, sharey=True)

        if self.channels == 1:
            ax = (ax,)

        [ax[i].plot(self.signal[:, i]) for i in range(self.channels)]
        fig.show()
