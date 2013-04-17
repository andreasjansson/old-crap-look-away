#!/usr/bin/python -u

import spectrogram
import os
import time
import sys
import glob
import job

def do_work(worker, data):
    notes, c, fv = spectrogram.get_training_example(data['path'])
    if fv is None:
        return

    name = os.path.splitext(os.path.basename(data['path']))[0]
    worker.store(name, {'fv': fv, 'notes': notes})
    worker.log('Analysed %s' % name)

if __name__ == '__main__':
    worker = job.Job(sys.argv[1])
    worker.run_worker(do_work)
