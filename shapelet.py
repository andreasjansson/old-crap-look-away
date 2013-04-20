def find_shapelet(data, min_len, max_len):
    candidates = generate_candidates(data, min_len, max_len)
    best_gain = 0
    best_shapelet = None

    for subseq in candidates:
        gain = check_candidate(data, subseq)
        if gain > best_gain:
            best_gain = gain
            best_shapelet = subseq

    return best_shapelet

def generate_candidates(data, min_len, max_len):
    pool = set()
    for length in xrange(min_len, max_len):
        for seq in data:
            pool |= subsequences(seq, length)
    return pool

def subsequences(seq, length):
    subseqs = set()
    for t in xrange(len(seq) - length):
        subseqs.add(tuple(seq[t : t + length]))
    return subseqs

def check_candidate(data, subseq):
    pass
