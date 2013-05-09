#!/usr/bin/python -u

import job
import makam
import cPickle
import os
import sys
import numpy as np
from shapelet import *

def random_weights(job_name):

    j = job.Job(job_name)

    data_job = job.Job('sequences')
    data = data_job.get_data()['data']
    data = data[0:500]
    training, test = job.cross_partition(data)

    all_classes = np.empty((0, 0))
    all_support = np.empty((0, len(training)))
    all_candidates = []

    min_len = 2
    max_len = 3
    k = 5

    nclasses = max([t[0] for t in training]) + 1

    for length in xrange(min_len, max_len):

        classes, support, candidates = get_subsequence_support(cands, length, nclasses, training)

        normalise_subsequence_support(support, training)

        support, candidates, seqs = get_pruned_candidates(classes, support, candidates)

        all_classes = classes
        all_support = np.vstack((all_support, support))
        all_candidates += map(tuple, candidates)

    #print len(all_candidates)

    classes = all_classes
    support = all_support
    candidates = all_candidates

    score = 0

    all_actual = []
    all_predicted = []

    nfeatures = support.shape[0]

    def classify(w):
        predicted = []
        for d in test:
            predicted.append(classify_knn(classes, support, candidates, d[1], k, w))

        actual = np.array([d[0] for d in test])
        score = np.sum(predicted == actual)
        return score

    reference_score = classify(np.array([1] * nfeatures)) / float(len(test))

    print '%d: reference: %.2f' % (job.INDEX(), reference_score)

    best_score = 0
    best_weights = None
    for i in np.arange(400):
        w = np.random.random(nfeatures)
        score = classify(w) / float(len(test))
        print '%d: %.2f, %.2f' % (job.INDEX(), score, score - reference_score)

        if score > best_score:
            best_score = score
            best_weights = w

    j.store_instance('reference_score', reference_score)
    j.store_instance('score', best_score)
    j.store_instance('weights', best_weights)

if __name__ == '__main__':
    random_weights(sys.argv[1])
