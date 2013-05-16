#!/usr/bin/python -u

import job
import shapelet
import makam
import cPickle
import os
import sys
import numpy as np

CACHE_TRAINING = False

def crossvalidate_new(job_name):

    j = job.Job(job_name)

    data_job = job.Job('sequences2')
    data = data_job.get_data('data')[0]
    training, test = job.cross_partition(data)

    #weights = data_job.get_data()['weights']

    predicted, actual, score = shapelet.knn_accuracy(training, test, 1, 4, 25)
    score *= 100.

    nclasses = max([d[0] for d in training]) + 1

    confusion = np.zeros((nclasses, nclasses))
    for p, a in zip(predicted, actual):
        confusion[p, a] += 1

    j.store_instance('score', score)
    j.store_instance('confusion', confusion)
    j.log('%d' % score)


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
        candidates = shapelet.generate_candidates(training, 3, 4)
        classes, subsequence_support, candidates = shapelet.get_subsequence_support(candidates, 3, 14, training)
        candidates = shapelet.get_pruned_candidates(classes, subsequence_support, np.array(candidates))
        
        with open(cache_filename, 'w') as f:
            cPickle.dump(candidates, f)

    candidates = shapelet.normalise_candidates(candidates)

    predicted = np.array([np.argmax(shapelet.classify(candidates, x[1], 14)) for x in test])
    actual = np.array([x[0] for x in test])
    score = 100 * sum(predicted == actual) / float(len(test))

    j.store_instance('score', score)
    j.store_instance('predicted', predicted)
    j.store_instance('actual', actual)
    j.log('%d' % score)

if __name__ == '__main__':
    crossvalidate_new(sys.argv[1])
