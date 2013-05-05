import numpy as np
import scipy.weave
import os
import math
import random

class Candidate(object):
    def __init__(self, seq, cls, example):
        self.seq = seq
        self.cls = cls
        self.example = example

def generate_candidates(data, min_len, max_len):
    filtered_candidates = []
    nclasses = max([d[0] for d in data]) + 1
    for length in xrange(min_len, max_len):
        candidates = {}
        for i, (cls, seq) in enumerate(data):
            find_subsequences(candidates, seq, length, cls, i)
    return candidates

#        classes, subsequence_support = get_subsequence_support(candidates, length, nclasses, data)
#        filtered_candidates += get_pruned_candidates(classes, subsequence_support, np.array(candidates.keys()))
#    return filtered_candidates

_subsequence_support_code = None
def get_subsequence_support(seq_candidates, seq_len, nclasses, data):
    subsequence_support = np.zeros((len(seq_candidates), len(data)))
    candidate_matrix = np.array(seq_candidates.keys())
    candidates = []
    for seq in candidate_matrix:
        candidates.append(seq_candidates[tuple(seq)])

    global _subsequence_support_code
    if _subsequence_support_code is None:
        source_filename = os.path.dirname(os.path.realpath(__file__)) + '/shapelet_subsequence_support.c'
        with open(source_filename, 'r') as f:
            _subsequence_support_code = f.read()

    len_seq_candidates = len(seq_candidates)
    total_examples = len(data)

    scipy.weave.inline(
        _subsequence_support_code,
        ['len_seq_candidates', 'candidate_matrix',
         'candidates', 'seq_len', 'subsequence_support', 'total_examples'])

    example_classes = np.array([d[0] for d in data])
    return example_classes, subsequence_support, candidate_matrix

def normalise_subsequence_support(subsequence_support, data):
    example_lengths = [float(len(d[1])) for d in data]
    subsequence_support /= example_lengths

def get_pruned_candidates(classes, subsequence_support, candidate_matrix):
    ncands, nclasses = subsequence_support.shape
    class_matrix = np.zeros((ncands, nclasses))

    for row in xrange(ncands):
        class_means = np.zeros((nclasses))
        for i in range(nclasses):
            class_examples = subsequence_support[row, classes == i]
            if not len(class_examples):
                continue
            class_means[i] = power_mean(class_examples)
        class_matrix[row, :] = class_means

    support_threshold = 1
    seqs = []

    for i, row in enumerate(class_matrix):
        if max(row) > support_threshold:
            seqs.append((candidate_matrix[i], row))

    return seqs

def classify_old(cands, seq, nclasses):
    seq = downsample(seq)
    class_prob = np.zeros((nclasses))
    
    for cand in cands:
        for i in xrange(0, len(seq) - len(cand[0])):
            if np.all(seq[i:i + len(cand[0])] == cand[0]):
                class_prob += cand[1] * len(cand[0])

    print np.argmax(class_prob)
    return class_prob


def classify(cands, seq, nclasses):
    seq = downsample(seq)
    seq_len = len(seq)
    cands_len = len(cands)
    class_prob = np.zeros((nclasses))
    cand_seqs = [c[0] for c in cands]
    cand_seq_lens = np.array([len(c[0]) for c in cands])
    cand_probs = [c[1] for c in cands]

    source_filename = os.path.dirname(os.path.realpath(__file__)) + '/classify.c'
    scipy.weave.inline(
        open(source_filename).read(),
        ['cands', 'cands_len', 'cand_seqs', 'cand_seq_lens',
         'seq', 'seq_len', 'cand_probs', 'nclasses', 'class_prob'])

    return class_prob

def classify_knn(classes, support, cands, seq):
    seq = downsample(seq)
    seq_support = np.zeros(len(cands))
    
    for c, cand in enumerate(cands):
        for i in xrange(0, len(seq) - len(cand)):
            if np.all(seq[i:i + len(cand)] == cand):
                seq_support[c] += 1

    seq_support /= float(len(seq))

    distances = np.zeros(support.shape[1])
    for i, example_support in enumerate(support.T):
        distances[i] = np.linalg.norm(example_support - seq_support)

    return distances

def downsample(seq):
    seq = np.round(np.array(seq) * 24 / 53.).astype(int)
    return tuple(seq)

def find_subsequences(candidates, seq, length, cls, i):
    for t in xrange(len(seq) - length):
        subseq = seq[t : t + length]
        candidate = Candidate(subseq, cls, i)
        subseq = downsample(subseq)
        if subseq in candidates:
            candidates[subseq].append(candidate)
        else:
            candidates[subseq] = [candidate]

def information_gain(data, data1, data2):
    entropy1 = entropy(data1) * len(data1) / len(data)
    entropy2 = entropy(data2) * len(data2) / len(data)
    return entropy(data) - (entropy1 + entropy2)

def entropy(classes):
    classes = np.array(classes)
    class_hist = np.bincount(classes) / float(len(classes))
    if np.all(classes[0] == classes):
        return 0
    lg = np.log(class_hist)
    lg[np.isinf(lg)] = 0
    ent = 0 - np.sum(class_hist * lg)
    return ent

def entropy2(values):
    values = values / float(sum(values))
    lg = np.log(values)
    lg[np.isinf(lg)] = 0
    ent = 0 - np.sum(values * lg)
    return ent
    

def test_data(by_makam):
    makams = by_makam.keys()
    #makams = random.sample(makams, 14)
    data = [d for m in makams for d in by_makam[m][:]]

    sequences = []
    for d in data:
        makam = makams.index(d['makam'])
        seq = np.mod(d['notes'][:,2].astype(int), 53)
        sequences.append((makam, seq))

    return sequences

def normalise_candidates(candidates):
    cand_matrix = np.array([c[1] for c in candidates])
    colsums = cand_matrix.sum(0)
    for i, c in enumerate(candidates):
        candidates[i] = (c[0], c[1] / colsums)
    return candidates

def power_mean(x, p=1.):
    x = np.power(x, p)
    return np.power(sum(x) / float(len(x)), 1./p)

def to_features(data, candidates):
    source_filename = os.path.dirname(os.path.realpath(__file__)) + '/shapelet_to_features.c'
    with open(source_filename, 'r') as f:
        _to_features_code = f.read()

    classes, seqs = zip(*data)
    cands = [c[0] for c in candidates]

    seqs = map(lambda s: np.array(downsample(s)), seqs)

    occurrences = np.zeros((len(seqs), len(cands)))

    scipy.weave.inline(
        _to_features_code,
        ['seqs', 'cands', 'occurrences'])

    return classes, occurrences

