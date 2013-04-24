import numpy as np
import scipy.weave
import os
import math

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

# data = zip(classes, examples)
def build_decision_tree(data, min_len, max_len, max_entropy=0):
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
#   parallellise filter candidates inner loop
#   "downsample" or reduce dimensionality, e.g. by using 24 bins instead of 53
#   keep reference from downsampled sequence back to original
#   euclidean distance doesn't work for music, use other distance (e.g. number of different notes)
#   use symbolic representation to determine how much more work needs to be done on pitch detector
def generate_candidates(data, min_len, max_len):
    filtered_candidates = []
    nclasses = max([d[0] for d in data]) + 1
    for length in xrange(min_len, max_len):
        candidates = {}
        seq = downsample(seq)
        for cls, seq in data:
            find_subsequences(candidates, seq, length, cls)
        filtered_candidates += filter_candidates(candidates, length, nclasses)
    return filtered_candidates

def filter_candidates(candidates, candidate_len, nclasses):
    class_matrix = np.zeros((len(candidates), nclasses))
    candidate_matrix = np.array(candidates.keys())

    print len(candidates)

    masking_iterations = 10
    nmask = int(math.ceil(candidate_len / 2.5))
    for iteration in xrange(masking_iterations):
        print iteration
        m = candidate_matrix.copy()
        mask_indices = np.random.choice(candidate_len, nmask, replace=False)
        m[:, mask_indices] = -1

        for c in xrange(len(candidates)):
            for seq, classes in candidates.iteritems():
                # tiny optimisation:
                if seq[0] != candidate_matrix[c, 0]:
                    continue

                if np.all(candidate_matrix[c, :] == seq):
                    for cls in classes:
                        class_matrix[c, cls] += 1

    row_sums = class_matrix.sum(axis=1).astype(float)
    class_matrix /= row_sums[:, np.newaxis]
    lg = np.log(class_matrix)
    class_entropy = 0 - np.sum(class_matrix * lg, axis=1)
    ordered_classes = np.argsort(class_entropy)

    nret = max(len(candidates) * .01, 3)
    return ordered_classes[:nret]

def find_subsequences(candidates, seq, length, cls):
    for t in xrange(len(seq) - length):
        subseq = tuple(seq[t : t + length])
        if subseq in candidates:
            candidates[subseq].append(cls)
        else:
            candidates[subseq] = [cls]

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
def subsequence_dist_new(seq, subseq):
    global _subsequence_dist_code
    if _subsequence_dist_code is None:
        source_filename = os.path.dirname(os.path.realpath(__file__)) + '/subsequence_dist.c'
        with open(source_filename, 'r') as f:
            _subsequence_dist_code = f.read()

    seq_len = len(seq);
    subseq_len = len(subseq)

    seq = np.array(seq)
    subseq = np.array(subseq)

    return scipy.weave.inline(
        _subsequence_dist_code,
        ['seq', 'subseq', 'seq_len', 'subseq_len'],
        type_converters=scipy.weave.converters.blitz)

def subsequence_dist(seq, subseq):
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
