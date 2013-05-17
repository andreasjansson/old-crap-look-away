#!/usr/bin/python -u

import job
import shapelet
import makam
import cPickle
import os
import sys
import numpy as np

CACHE_TRAINING = False

def crossvalidate_queue_do_work(worker, data):
    index = data['index']

    data_job = job.Job('sequences_arel')
    data = data_job.get_data('data')[0]
    training = data[:index] + data[(index + 1):]
    test = [data[index]]

    predicted, actual, score = shapelet.knn_accuracy(training, test, 1, 4, 15, 7)

    nclasses = max([d[0] for d in training]) + 1

    confusion = (predicted, actual)

    worker.store('score', score, index)
    worker.store('confusion', confusion, index)
    #worker.log('%d' % score)


def crossvalidate_queue(job_name):
    worker = job.Job(job_name)
    worker.run_worker(crossvalidate_queue_do_work)

def crossvalidate(job_name):

    j = job.Job(job_name)

    data_job = job.Job('sequences_arel')
    data = data_job.get_data('data')[0]
    training, test = job.cross_partition(data)

    predicted, actual, score = shapelet.knn_accuracy(training, test, 1, 4, 15, 5)
    score *= 100.

    nclasses = max([d[0] for d in training]) + 1

    confusion = np.zeros((nclasses, nclasses))
    for p, a in zip(predicted, actual):
        confusion[p, a] += 1

    j.store_instance('score', score)
    j.store_instance('confusion', confusion)
    j.log('%d' % score)



if __name__ == '__main__':
    crossvalidate_queue(sys.argv[1])
    #crossvalidate(sys.argv[1])
