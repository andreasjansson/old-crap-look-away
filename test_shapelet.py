#!/usr/bin/python -u

import job
import makam
import shapelet
import cPickle
import sys

if __name__ == '__main__':
    j = job.Job('notes')
    data = cPickle.load(open('data.pkl', 'r'))#j.get_data()
    #cPickle.dump(data, open('data.pkl', 'w'))
    d40 = makam.data_prune_tiny(data, 40)
    makams, data = makam.data_to_sequences(d40)
    tree = shapelet.build_decision_tree(data[:50], 3, 5)
    j = job.Job(sys.argv[1])
    j.store('tree', tree)
    print tree
