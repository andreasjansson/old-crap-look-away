import numpy as np

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

def build_decision_tree(data, min_len, max_len, max_entropy=0):
    classes = [d[0] for d in data]
    if entropy(classes) <= max_entropy:
        class_hist = np.bincount(classes)
        return DTLeaf(np.argmax(class_hist), classes)
    shapelet, split_point = find_shapelet(data, min_len, max_len)
    data1, data2 = split_by_shapelet(data, shapelet, split_point)
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

    for subseq in candidates:
        split_point, gain = check_candidate(data, subseq)
        if gain > best_gain:
            best_gain = gain
            best_shapelet = subseq
            best_split_point = split_point

    return best_shapelet, best_split_point

def generate_candidates(data, min_len, max_len):
    pool = set()
    for length in xrange(min_len, max_len):
        for cls, seq in data:
            pool |= subsequences(seq, length)
    return pool

def subsequences(seq, length):
    subseqs = set()
    for t in xrange(len(seq) - length):
        subseqs.add(tuple(seq[t : t + length]))
    return subseqs

def check_candidate(data, subseq):
    hist = {}
    for cls, seq in data:
        seq = np.array(seq)
        dist = subsequence_dist(seq, subseq)
        if dist in hist:
            hist[dist] += [cls]
        else:
            hist[dist] = [cls]

    return optimal_split_point(hist)

def subsequence_dist(seq, subseq):
    best_dist = float('inf')
    best_start = 0
    for start in xrange(0, len(seq) - len(subseq)):
        dist = sequence_dist(seq[start:start + len(subseq)], subseq)
        if dist < best_dist:
            best_dist = dist
            best_start = start
    return best_dist

def sequence_dist(seq1, seq2):
    return np.linalg.norm(seq1 - seq2)

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
    ent = 0 - np.sum(class_hist * np.log(class_hist))
    return ent
