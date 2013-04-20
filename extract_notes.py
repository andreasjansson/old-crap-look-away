#!/usr/bin/python -u

import audio
import spectrogram
import os
import time
import sys
import glob
import job
from makam import *

def do_work(worker, data):
    window_size = 8192

    a = audio.Audio(data['path'])
    s = spectrogram.Spectrogram(a.signal[:, 0], a.sample_rate, window_size, 2048)
    cls = class_from_filename(data['path'])

    # not expecting any F0 above 2500 Hz, speeds things up
    path = monophonic_path(s.data[0:500,:])
    notes = notes_from_path(path)
    notes = notes_to_pitches(notes, 53, a.sample_rate, window_size)

    name = os.path.splitext(os.path.basename(data['path']))[0]
    worker.store(name, {'notes': notes, 'makam': cls})
    worker.log('Analysed %s' % name)

if __name__ == '__main__':
    worker = job.Job(sys.argv[1])
    worker.run_worker(do_work)
