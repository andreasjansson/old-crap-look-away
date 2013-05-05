#!/usr/bin/python -u

import job
import shapelet
import makam
import cPickle
import os
import sys
import numpy as np

CACHE_TRAINING = False

def crossvalidate(job_name):

    j = job.Job(job_name)

    data_job = job.Job('sequences')
    data = data_job.get_data()['data']
    training, test = job.cross_partition(data)

    cache_filename = 'training_cache_%d_%d.pkl' % (job.INDEX(), job.COUNT())

    if CACHE_TRAINING and os.path.exists(cache_filename):
        with open(cache_filename, 'r') as f:
            candidates = cPickle.load(f)
    else:
        j.log('Staring training %d/%d' % (job.INDEX() + 1, job.COUNT()))
        candidates = shapelet.generate_candidates(training, 3, 8)
        with open(cache_filename, 'w') as f:
            cPickle.dump(candidates, f)

    candidates = shapelet.normalise_candidates(candidates)

    predicted = np.array([np.argmax(shapelet.classify(candidates, x[1], 14)) for x in test])
    actual = np.array([x[0] for x in test])
    score = 100 * sum(predicted == actual) / float(len(test))

    j.store('score_%d_%d' % (job.INDEX(), job.COUNT()), score)
    j.store('predicted_%d_%d' % (job.INDEX(), job.COUNT()), predicted)
    j.store('actual_%d_%d' % (job.INDEX(), job.COUNT()), actual)
    j.log('%d' % score)

if __name__ == '__main__':
    crossvalidate(sys.argv[1])
