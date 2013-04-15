#!/usr/bin/python

import spectrogram
import cPickle
import os
import time
import sys
import glob
import job

def do_work(worker, data):
    c, fv = spectrogram.get_training_example(data['path'])
    name = os.path.splitext(os.path.basename(data['path']))[0]
    fv = cPickle.dumps(fv)
    worker.store(name, fv)
    worker.log('Analysed %s' % name)

if __name__ == '__main__':
    worker = job.Job(sys.argv[1])
    worker.run_worker(do_work)
