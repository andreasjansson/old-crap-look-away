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

def generate_candidates(data, length):
    filtered_candidates = []
    nclasses = max([d[0] for d in data]) + 1
    candidates = {}
    for i, (cls, seq) in enumerate(data):
        find_subsequences(candidates, seq, length, cls, i)
    return candidates

_subsequence_support_code = None
def get_subsequence_support(seq_candidates, data):
    seq_len = len(seq_candidates.keys()[0])
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
    ncands = subsequence_support.shape[0]
    nclasses = max(classes) + 1
    class_matrix = np.zeros((ncands, nclasses))

    for row in xrange(ncands):
        class_means = np.zeros((nclasses))
        for i in range(nclasses):
            class_examples = subsequence_support[row, classes == i]
            if not isinstance(class_examples, np.ndarray):
                class_examples = np.array([class_examples])
            if not len(class_examples):
                continue
            class_means[i] = power_mean(class_examples)
        class_matrix[row, :] = class_means

    support_threshold = .00001
    seqs = []

    indices = []
    for i, row in enumerate(class_matrix):
        if max(row) > support_threshold:
            indices.append(i)
            seqs.append((candidate_matrix[i], row))

    indices = np.array(indices)
    return subsequence_support[indices], candidate_matrix[indices], seqs

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

_seq_support_code = None
def get_seq_support(cands, seq):
    global _seq_support_code
    if _seq_support_code is None:
        source_filename = os.path.dirname(os.path.realpath(__file__)) + '/shapelet_seq_support.c'
        _seq_support_code = open(source_filename, 'r').read()

    assert isinstance(cands, list)
    assert isinstance(cands[0], tuple)

    seq = downsample(seq.astype(int))
    
    seq_len = len(seq)
    cands_len = len(cands)
    cand_seq_lens = np.array([len(c) for c in cands])
    seq_support = np.zeros(len(cands))

    try:
        scipy.weave.inline(
            _seq_support_code,
            ['cands', 'cands_len', 'cand_seq_lens', 'seq', 'seq_len', 'seq_support'])
    except TypeError, e:
        print 'cands_len: ', cands_len
        print 'seq_len: ', seq_len
        print e

    return seq_support

def weigh_near(nearest):
    k = len(nearest)
    nearest_map = {}
    near_weight = 1
    for i, cls in enumerate(nearest):
        if cls not in nearest_map:
            nearest_map[cls] = 0
        nearest_map[cls] += 1 + np.power(near_weight, (k - i + 1) / float(k))
    return nearest_map

def get_local_radii(support, classes):
    radii = np.zeros((support.shape[1]))
    support_t = support.T
    for i, s in enumerate(support_t):
        dists = np.abs(s - support_t).sum(1)
        near = np.argsort(dists)[1:]
        closest_same = near[classes[near] == classes[i]][0]
        radii[i] = dists[closest_same]
    return radii

def get_local_weights(support, classes, k=10):
    local_weights = np.zeros(support.shape) + 1.
    support_t = support.T
    for i, s in enumerate(support_t):
        near = np.argsort(np.abs(s - support_t).sum(1))[1:]
        closest_same = near[classes[near] == classes[i]][0]
        closest_other = near[:k][classes[near[:k]] != classes[i]]
        if not len(closest_other):
            continue
        closest_other = closest_other
        w = 1 + np.abs(s - support[:, closest_other].T).sum(0) - np.abs(s - support[:, closest_same])
        local_weights[:,i] = w
    return local_weights

def classify_knn(classes, support, cands, seq, k):
    seq_support = get_seq_support(cands, seq)

    seq_support /= float(len(seq))

    distances = np.zeros(support.shape[1])

    for i, example_support in enumerate(support.T):
        distances[i] = np.power(np.sum(np.power(np.abs(example_support - seq_support), 2.)), 1/2.)

    nearest = classes[np.argsort(distances)[::1]][:k]
    nearest_map = weigh_near(nearest)

    #print zip(np.argsort(distances), nearest), {k: int(v) for k, v in nearest_map.iteritems()}


    return max(nearest_map, key=nearest_map.get)

def downsample(seq):
    # disabled temporarily while dealing with symbolic data
    seq = np.mod(np.array(seq), 53.).astype(int)
    #seq = np.round(np.array(seq) * 12 / 53.).astype(int)
    return tuple(seq)

def find_subsequences(candidates, seq, length, cls, i):
    for t in xrange(len(seq) - length):
        subseq = seq[t : t + length]
        if any([math.isnan(x) for x in subseq]):
            continue

        candidate = Candidate(subseq, cls, i)
        subseq = downsample(subseq)
        if subseq in candidates:
            candidates[subseq].append(candidate)
        else:
            candidates[subseq] = [candidate]

def information_gain(data, data1, data2):
    if len(data1) == 0 or len(data2) == 0:
        return 0
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

def power_mean(x, p=.5):
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

def sum_support_by_class(support, classes):
    nclasses = max(classes) + 1
    nperclass = np.bincount(classes)
    cls_support = np.zeros((support.shape[0], nclasses))
    for i, c in enumerate(classes):
        cls_support[:, c] += support[:, i]
    cls_support /= nperclass
    return cls_support

def knn_accuracy(training, test, min_len, max_len, k):
    global all_classes, all_support, all_candidates

    all_classes = np.empty((0, 0))
    all_support = np.empty((0, len(training)))
    all_candidates = []

    nclasses = max([t[0] for t in training]) + 1

    for length in xrange(min_len, max_len):

        cands = generate_candidates(training, length)
        classes, support, candidates = get_subsequence_support(cands, training)
        normalise_subsequence_support(support, training)

        support, candidates, seqs = get_pruned_candidates(np.array(classes), support, candidates)

        all_classes = classes
        all_support = np.vstack((all_support, support))
        all_candidates += map(tuple, candidates)

        #print 'length: %d, candidates: %d, total candidates: %d' % (length, len(candidates), len(cands))

    #print len(all_candidates)

    classes = all_classes
    support = all_support
    candidates = all_candidates

    score = 0

    all_actual = []
    all_predicted = []

    #predicted = np.array([classify_knn(classes, support, candidates, d[1], k) for d in test])

    predicted = []
    for d in test:
        predicted.append(classify_knn(classes, support, candidates, d[1], k))
        #print d[0]

    actual = np.array([d[0] for d in test])

    score = np.sum(predicted == actual)

    return predicted, actual, score / float(len(test))

def test_random_weights(support, classes, weights):

    global g_weights
    g_weights = weights

    support = (support.T * weights).T

    def dist(m1, m2):
        return np.sum([np.abs(m1.T - m2.T[i]).sum() for i in np.arange(m2.shape[1])]) / float(m2.shape[1])

    cs = np.unique(classes)
    dists = np.zeros((len(cs), len(cs)))

    scores = np.zeros((len(cs)))

    for i in np.unique(classes):
        this = support[:, classes == i]
        for j in np.unique(classes):
            other = support[:, classes == j]

            d = dist(this, other)
            dists[i, j] = d / this.shape[1]

    score = dists.sum()

    import matplotlib.pyplot as plt
    plt.imshow(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(support.T[np.argsort(classes)], 'cityblock')), interpolation='nearest')

    print dists.sum() / dists.diagonal().sum()

    return dists

def cor(support):
    return scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(support.T, 'cityblock'))


def classify_dt(dt, seq, cands):
    node = dt
    while not hasattr(node, 'cls'): # isinstance DTLeaf doesn't seem to work???
        sup = (get_seq_support([tuple(node.shapelet)], seq) / float(len(seq)))[0]
        if sup < node.split_point:
            node = node.left
        else:
            node = node.right

    return node.cls

def build_dt(support, classes, cands):

    print len(classes), entropy(classes), classes_str(classes)

    if entropy(classes) < 0.5 or len(classes) < 15:
        return DTLeaf(classes)

    # prune support
    cands_i, value = optimal_split(support, classes)
    indices1 = support[cands_i] <= value
    indices2 = support[cands_i] > value
    classes1 = classes[indices1]
    classes2 = classes[indices2]

    return DTNode(cands[cands_i], value, classes,
                  build_dt(support.T[indices1].T, classes1, cands),
                  build_dt(support.T[indices2].T, classes2, cands))


def optimal_split(support, classes, steps=4):
    best_gain = 0
    best_split = None
    for i in xrange(len(support)):
        s = support[i,:]
        low = np.min(s)
        high = np.max(s)
        for x in np.linspace(low, high, steps)[1:-1]:
            classes1 = classes[s <= x]
            classes2 = classes[s > x]
            gain = information_gain(classes, classes1, classes2)
            if gain > best_gain:
                best_gain = gain
                best_split = (i, x)
    return best_split

def classify_dt_by_order(dt, seq, cands):
    node = dt

    while not hasattr(node, 'cls'): # isinstance DTLeaf doesn't seem to work???
        sup = (get_seq_support([tuple(node.shapelet1), tuple(node.shapelet2)], seq) / float(len(seq)))
        if sup[0] <= sup[1]:
            node = node.left
        else:
            node = node.right

    return node.cls

def build_dt_by_order(support, classes, cands):

    print len(classes), entropy(classes), classes_str(classes)

    if entropy(classes) < 0.3 or len(classes) < 15:
        return DTLeaf(classes)

    # prune support
    first_i, second_i = optimal_split_by_order(support, classes)
    indices1 = support[first_i] <= support[second_i]
    indices2 = support[first_i] > support[second_i]
    classes1 = classes[indices1]
    classes2 = classes[indices2]

    return DTNodeByOrder(cands[first_i], cands[second_i], classes,
                  build_dt_by_order(support.T[indices1].T, classes1, cands),
                  build_dt_by_order(support.T[indices2].T, classes2, cands))

def optimal_split_by_order(support, classes):
    best_gain = 0
    best_split = None
    for i in xrange(len(support)):
        for j in xrange(len(support)):
            classes1 = classes[support[i] <= support[j]]
            classes2 = classes[support[i] > support[j]]
            gain = information_gain(classes, classes1, classes2)
            if gain > best_gain:
                best_gain = gain
                best_split = (i, j)
    return best_split



def classes_str(classes):
    b = np.bincount(classes)
    bw = np.where(b > 0)[0]
    return str(dict(zip(list(bw), list(b[bw]))))

class DTNode(object):
    def __init__(self, shapelet, split_point, classes, left=None, right=None):
        self.shapelet = list(shapelet)
        self.split_point = split_point
        self.classes = classes
        self.left = left
        self.right = right

    def print_node(self, level=0):
        before = '| ' * (level)
        if level > 0:
            before = before + '\n' + '| ' * (level - 1) + '+-'
        s = '%s%s (%.3f) %s\n' % (before, str(self.shapelet),
                                       entropy(self.classes), classes_str(self.classes))
        s += self.left.print_node(level + 1)
        s += self.right.print_node(level + 1)
        return s

    def __str__(self):
        return self.print_node()

    def __repr__(self):
        return self.__str__()

class DTNodeByOrder(DTNode):
    def __init__(self, shapelet1, shapelet2, classes, left=None, right=None):
        self.shapelet1 = list(shapelet1)
        self.shapelet2 = list(shapelet2)
        self.classes = classes
        self.left = left
        self.right = right

    def print_node(self, level=0):
        before = '| ' * (level)
        if level > 0:
            before = before + '\n' + '| ' * (level - 1) + '+-'
        s = '%s%s <= %s (%.3f) %s\n' % (before, str(self.shapelet1), str(self.shapelet2),
                                       entropy(self.classes), classes_str(self.classes))
        s += self.left.print_node(level + 1)
        s += self.right.print_node(level + 1)
        return s


class DTLeaf(object):
    def __init__(self, classes):
        self.classes = classes
        self.cls = np.argmax(np.bincount(classes))

    def print_node(self, level=0):
        before = '| ' * level + '\n' + '| ' * (level - 1) + '+-> '
        return '%s%s\n' % (before, str(self))

    def __str__(self):
        return '%s (%.3f) %s' % (self.cls, entropy(self.classes), classes_str(self.classes))

    def __repr__(self):
        return self.__str__()


def confusion_matrix(actual, predicted):
    nclasses = max(np.max(actual), np.max(predicted)) + 1
    confusion = np.zeros((nclasses, nclasses), dtype=int)
    for p, a in zip(predicted, actual):
        confusion[p, a] += 1
    return confusion

def random_binary_weights(support, classes, test, cands):
    pass


def within_class_covariance_matrix(support, classes):
    mat = np.zeros((len(classes), len(classes)))
    for i, c in enumerate(classes):
        pass

def lda(support, classes):
    between_class = np.cov(sum_support_by_class(support, classes))
    within_class = within_class_covariance_matrix(support, classes)
    
    
