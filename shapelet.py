import numpy as np
import scipy.weave
import os
import math
import random

def test_weave():
    a = [Candidate(np.array([0,10,20,30]), 0), Candidate(np.array([0,40,50,60]), 1), Candidate(np.array([0,70,80,90]), 2)]
    b = np.arange(12).reshape(4,3)
    c = np.zeros((4, 3))
    scipy.weave.inline(open('test.c', 'r').read(), ['a', 'b', 'c'])

    return c

class Candidate(object):
    def __init__(self, seq, cls, example):
        self.seq = seq
        self.cls = cls
        self.example = example

class DTNode(object):
    def __init__(self, shapelet, split_point, classes, left, right):
        self.shapelet = list(shapelet)
        self.split_point = split_point
        self.classes = classes
        self.left = left
        self.right = right

    def print_node(self, level=0):
        before = '| ' * (level - 1)
        if level > 0:
            before = before + '| ' * level + '\n' + before + '+-'
        s = '%s%s (%.3f)\n' % (before, str(self.shapelet),
                                       entropy(self.classes))
        s += self.left.print_node(level + 1)
        s += self.right.print_node(level + 1)
        return s

    def __str__(self):
        return self.print_node()

    def __repr__(self):
        return self.__str__()

class DTLeaf(object):
    def __init__(self, cls, classes):
        self.cls = cls
        self.classes = classes

    def print_node(self, level=0):
        before = '| ' * level + '\n' + '| ' * (level - 1) + '+-> '
        return '%s%s\n' % (before, str(self))

    def __str__(self):
        return '%s (%.3f)' % (self.cls, entropy(self.classes))

    def __repr__(self):
        return self.__str__()

def shapelet_accuracy(data, training_ratio = .7):
    random.shuffle(data)
    split = int(len(data) * training_ratio)
    training = data[:split]
    test = data[split:]
    dt = build_decision_tree(training)
    
def classify(dt, example):
    node = dt
    while not hasattr(node, 'cls'): # isinstance DTLeaf doesn't seem to work???
        dist = subsequence_dist(example, node.shapelet)
        if dist < node.split_point:
            node = node.left
        else:
            node = node.right

    return node.cls

# data = zip(classes, examples)
def build_decision_tree(data, min_len, max_len, max_entropy=0.2):
    classes = [d[0] for d in data]
    if entropy(classes) <= max_entropy:
        class_hist = np.bincount(classes)
        return DTLeaf(np.argmax(class_hist), classes)
    shapelet, split_point = find_shapelet(data, min_len, max_len)
    data1, data2 = split_by_shapelet(data, shapelet, split_point)

    if len(data1) == 0 or len(data2) == 0:
        import IPython
        IPython.embed()

    return DTNode(shapelet, split_point, classes,
                  build_decision_tree(data1, min_len, max_len, max_entropy),
                  build_decision_tree(data2, min_len, max_len, max_entropy))

def split_by_shapelet(data, shapelet, split_point=None):
    if split_point is None:
        split_point, gain = check_candidate(data, shapelet)
    data1 = []
    data2 = []

    for cls, seq in data:
        seq = np.array(seq)

        if subsequence_dist(seq, shapelet) < split_point:
            data1.append((cls, seq))
        else:
            data2.append((cls, seq))
    return data1, data2

def find_shapelet(data, min_len, max_len):
    candidates = generate_candidates(data, min_len, max_len)
    best_gain = 0
    best_shapelet = None
    best_split_point = None

    i = 0
    for subseq in candidates:
        i += 1
        print '%d/%d' % (i , len(candidates))
        split_point, gain = check_candidate(data, subseq, best_gain)
        if gain > best_gain:
            best_gain = gain
            best_shapelet = subseq
            best_split_point = split_point

    return best_shapelet, best_split_point

# TODO: UPNEXT:
#   use symbolic representation to determine how much more work needs to be done on pitch detector
#   euclidean distance doesn't work for music, use other distance (e.g. number of different notes)
#   parallellise filter candidates inner loop
def generate_candidates(data, min_len, max_len):
    filtered_candidates = []
    nclasses = max([d[0] for d in data]) + 1
    for length in xrange(min_len, max_len):
        candidates = {}
        for i, (cls, seq) in enumerate(data):
            find_subsequences(candidates, seq, length, cls, i)
        filtered_candidates += filter_candidates(candidates, length, nclasses, data)
    return filtered_candidates

_filter_candidates_code = None
def filter_candidates(seq_candidates, seq_len, nclasses, data):
    class_matrix = np.zeros((len(seq_candidates), nclasses))
    example_matrix = np.zeros((len(seq_candidates), len(data)))
    candidate_matrix = np.array(seq_candidates.keys())
    candidates = []
    for seq in candidate_matrix:
        candidates.append(seq_candidates[tuple(seq)])
    #candidate_classes = np.array(candidate_classes)

    masking_iterations = 1
    nmask = 0 * int(math.ceil(seq_len / 2.5))

    global _filter_candidates_code
    if _filter_candidates_code is None:
        source_filename = os.path.dirname(os.path.realpath(__file__)) + '/shapelet_filter_candidates.c'
        with open(source_filename, 'r') as f:
            _subsequence_dist_code = f.read()

    len_seq_candidates = len(seq_candidates)

    for iteration in xrange(masking_iterations):
        masked_candidate_matrix = candidate_matrix.copy()
        mask_indices = np.random.choice(seq_len, nmask, replace=False)
        masked_candidate_matrix[:, mask_indices] = -1

        total_classes = class_matrix.shape[1]
        total_examples = len(data)

        scipy.weave.inline(
            _subsequence_dist_code,
            ['len_seq_candidates', 'masked_candidate_matrix',
             'candidates', 'class_matrix', 'seq_len', 'total_classes',
             'example_matrix', 'total_examples'])

    example_classes = np.array([d[0] for d in data])
    for row in xrange(len_seq_candidates):
        class_medians = np.zeros((nclasses))
        for i in range(nclasses):
            class_medians[i] = np.mean(example_matrix[row, example_classes == i])
        class_matrix[row, :] = class_medians

    support_threshold = 1
    seqs = []

    for i, row in enumerate(class_matrix):
        if max(row) > support_threshold:
            seqs.append((candidate_matrix[i], row))

    print len(seqs)

    return seqs

    row_sums = class_matrix.sum(axis=1).astype(float)
    norm_class_matrix = class_matrix / row_sums[:, np.newaxis]
    lg = np.log(norm_class_matrix)
    lg[np.isinf(lg)] = 0
    class_entropy = 0 - np.sum(norm_class_matrix * lg, axis=1)
    class_entropy[np.isnan(class_entropy)] = 0

    entropies = np.sort(np.unique(class_entropy))
    current_entropy_i = 0
    seq_indices = []
    count_threshold = 5
    while 1:
        indices = np.where(class_entropy == entropies[current_entropy_i])[0]
        import pdb
        pdb.set_trace()
        seq_indices = indices
        break

    downsampled_seqs = candidate_matrix[seq_indices, :]

    seqs = set()
    for seq in downsampled_seqs:
        seq_seqs = [tuple(c.seq) for c in seq_candidates[tuple(seq)]]
        seqs |= set(seq_seqs)

    return seqs

def test_classify_old(cands, seq, nclasses):
    seq = downsample(seq)
    class_prob = np.zeros((nclasses))
    
    for cand in cands:
        for i in xrange(0, len(seq) - len(cand[0])):
            if np.all(seq[i:i + len(cand[0])] == cand[0]):
                class_prob += cand[1]
                if cand[1][2] > cand[1][11]:
                    print '++++++ %s: %.2f, %.2f, %.2f, %.2f' % (str(cand[0]), cand[1][2], cand[1][11], sum(cand[1]), entropy2(cand[1]))
                else:
                    print '------ %s: %.2f, %.2f, %.2f, %.2f' % (str(cand[0]), cand[1][2], cand[1][11], sum(cand[1]), entropy2(cand[1]))

    print np.argmax(class_prob)
    return class_prob


def test_classify(cands, seq, nclasses):
    seq = downsample(seq)
    seq_len = len(seq)
    cands_len = len(cands)
    class_prob = np.zeros((nclasses))
    cand_seqs = [c[0] for c in cands]
    cand_seq_lens = np.array([len(c[0]) for c in cands])
    cand_probs = [c[1] for c in cands]

    scipy.weave.inline(
        open('test_classify.c').read(),
        ['cands', 'cands_len', 'cand_seqs', 'cand_seq_lens',
         'seq', 'seq_len', 'cand_probs', 'nclasses', 'class_prob'])

    print class_prob

    return class_prob
    
    for cand in cands:
        for i in xrange(0, len(seq) - len(cand[0])):
            if np.all(seq[i:i + len(cand[0])] == cand[0]):
                class_prob += cand[1]
                print cand
    print np.argmax(class_prob)
    return class_prob

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

def check_candidate(data, subseq, best_gain):
    hist = {}

    classes = [d[0] for d in data]
    class_hist = np.bincount(classes)

    for cls, seq in data:

        seq = np.array(seq)
        dist = subsequence_dist(seq, subseq)
        if dist in hist:
            hist[dist] += [cls]
        else:
            hist[dist] = [cls]

        class_hist[cls] -= 1
        if entropy_early_prune(best_gain, hist, class_hist):
            return 0, 0

    return optimal_split_point(hist)

def entropy_early_prune(best_gain, hist, class_hist):
    #return False
    hist = hist.copy()
    dists = hist.keys()
    min_dist = min(dists)
    max_dist = max(dists)
    use_min = True

    for i in np.nonzero(class_hist)[0]:
        if use_min:
            hist[min_dist] += [i] * class_hist[i]
        else:
            if max_dist + 1 in hist:
                hist[max_dist + 1] += [i] * class_hist[i]
            else:
                hist[max_dist + 1] = [i] * class_hist[i]
        use_min = not use_min

    split, gain = optimal_split_point(hist)
    return gain < best_gain


_subsequence_dist_code = None
def subsequence_dist(seq, subseq):
    global _subsequence_dist_code
    if _subsequence_dist_code is None:
        source_filename = os.path.dirname(os.path.realpath(__file__)) + '/subsequence_dist.c'
        with open(source_filename, 'r') as f:
            _subsequence_dist_code = f.read()

    seq_len = len(seq)
    subseq_len = len(subseq)

    subseq = np.array(subseq)

    return scipy.weave.inline(
        _subsequence_dist_code,
        ['seq', 'subseq', 'seq_len', 'subseq_len'])

def subsequence_dist_new(seq, subseq):
    best_dist = float('inf')
    best_start = 0
    for start in xrange(0, len(seq) - len(subseq)):
        dist = np.linalg.norm(seq[start:start + len(subseq)] - subseq)
        if dist < best_dist:
            best_dist = dist
            best_start = start
    return best_dist

def optimal_split_point(hist):
    dists = sorted(hist.keys())
    best_gain = 0
    best_split = None
    for first, second in zip(dists[:-1], dists[1:]):
        mean = (first + second) / 2
        hist1, hist2 = split_hist(hist, mean)
        gain = information_gain(flatten(hist), flatten(hist1), flatten(hist2))
        if gain > best_gain:
            best_gain = gain
            best_split = mean
    return best_split, best_gain

def split_hist(hist, split_point):
    dists = np.array(sorted(hist.keys()))
    split_index = np.where(dists >= split_point)[0][0]
    dists1, dists2 = np.split(dists, [split_index])
    return {k: hist[k] for k in dists1}, {k: hist[k] for k in dists2}

def flatten(hist):
    return np.concatenate(hist.values())

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
    data = [d for m in makams for d in by_makam[m][0:80]]

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
