import numpy as np
import math
import scipy
import scipy.misc
import scipy.weave
import sklearn.tree
import os.path
import glob
import cPickle
import spectrogram

def monophonic_path(spectrogram_data):
    values = np.max(spectrogram_data) - spectrogram_data
    costs = np.zeros(spectrogram_data.shape)
    prev = np.zeros(spectrogram_data.shape)
    amp = amplitude(spectrogram_data)

    change_cost = .1
    silence_cost = 2

    costs[:, 0] = values[:, 0]

    height, length = values.shape

    source_filename = os.path.dirname(os.path.realpath(__file__)) + '/dynprog.c'
    with open(source_filename, 'r') as f:
        code = f.read()
    scipy.weave.inline(code, ['values', 'costs', 'prev', 'change_cost',
                              'amp', 'silence_cost', 'length', 'height',],
                       type_converters=scipy.weave.converters.blitz)

    path = [0] * values.shape[1]
    path[-1] = np.argmin(costs[:, -1])
    for t in reversed(xrange(len(path) - 1)):
        path[t] = prev[path[t + 1], t]

    return path

def monophonic_maxima_path(spectrogram_data):
    return spectrogram_data.argmax(0)

# notes is a matrix with fields [start, end, note]
def notes_from_path(path, threshold=1):
    notes = np.empty((0, 3))
    current = []
    start_time = 0
    for t, x in enumerate(path):
        if len(current):
            mean = sum(current) / float(len(current))
        else:
            mean = 0

        if abs(x - mean) > threshold:
            if mean > 0:
                notes = np.vstack((notes, [start_time, t, round(mean)]))
            current = []
            start_time = t

        if x > 0:
            current.append(x)

    notes = notes.astype(int)
    return notes

def notes_quantise_pitch_class(note_bins, octave_steps, sample_rate, window_size):
    notes = note_bins.astype(float)
    freqs = note_bins * sample_rate / window_size
    base = 261.63 # c, TODO: should probably be something less western
    pitches = np.log2(freqs / base) * octave_steps
    classes = np.mod(pitches.astype(int), octave_steps)
    classes = np.round(classes).astype(int)

    return classes

def notes_to_pitches(notes, octave_steps, sample_rate, window_size):
    notes = notes.copy()
    freqs = notes[:,2] * sample_rate / window_size
    base = 261.63 # c, TODO: should probably be something less western
    pitches = np.round(np.log2(freqs / base) * octave_steps)
    pitches = pitches.astype(int)
    notes[:,2] = pitches
    return notes

# assuming argmax ioi = beat
# returns matrix with columns [time, klang_1, klang_2, [...], klang_n]
def get_nklangs(notes, nbeats, n, threshold=.5):
    iois = notes[1:, 0] - notes[:-1, 0]
    beat = np.argmax(np.bincount(iois))
    bars = []
    bar = []
    phase = 0
    time = 0

    for i, ioi in enumerate(iois):
        if ioi > beat * threshold:
            bar.append((ioi, notes[i, 2]))

        if phase > beat * nbeats:
            if len(bar) > 0:
                bars.append((time, bar))
            bar = []
            phase -= beat * nbeats

        phase += ioi
        time += ioi

    nklangs = np.zeros((len(bars), n + 1))
    for i, (time, bar) in enumerate(bars):
        nklang = [x[1] for x in sorted(bar, reverse=True)[0:n]]
        if len(nklang) < n:
            nklang = [nklang[0]] * (n - len(nklang) + 1) + nklang[1:]
        nklangs[i,:] = [time] + nklang

    nklangs = nklangs.astype(int)

    return nklangs

def nklangs_to_feature_vector(nklangs, octave_steps):
    n = nklangs.shape[1] - 1
    values = (nklangs[:, 1:] * np.power(octave_steps, np.arange(n))).sum(1)
    vector = np.bincount(values.astype(int))
    #length = np.sum(([octave_steps] * n) * np.power(octave_steps, np.arange(n)))
    # closed form
    length = (octave_steps * (octave_steps ** n - 1) / (octave_steps - 1))
    return pad1d(vector, 0, length)

# TODO UPNEXT plt.hist of nklang components
def feature_vector_to_nklangs(fv, octave_steps, n):
    indices = filter(lambda i: fv[i] > 0, fv.argsort())
    indices.reverse()
    nklangs = np.zeros((len(indices), n + 1))
    for order, i in enumerate(indices):
        x = fv[i]
        nklangs[order, 0] = x
        for j in range(n):
            nklangs[order, j + 1] = i % octave_steps
            i = int(i / octave_steps)
    return nklangs

def amplitude(spectrogram_data, smooth_width=50):
    window = np.hanning(smooth_width)
    window /= window.sum()

    amp = np.sum(spectrogram_data[1:,:], 0) # ignore 0 index
    return np.convolve(amp, window, mode='same')

def pad1d(array, x, length):
    return np.append(array, [x] * (length - len(array)))

# import audio
# a = audio.Audio(filename)
# s = Spectrogram(a.signal[:, 0], a.sample_rate, 4096, 2048)
# path = monophonic_maxima_path(s.data)
# notes = notes_from_path(path)
# steps = 12
# classes = notes_quantise_pitch_class(notes[:, 2], steps, s.sample_rate, s.window_size)
# notes[:,2] = classes
# nklangs = get_nklangs(notes, 4, 2)
# fv = nklangs_to_feature_vector(nklangs, steps)

# when using 55 steps per octave,
# we need a high window size, e.g. 8192, with shorter hop, e.g. 2048
# one way is to see how the chroma converges, by taking euclidean
# distance of consecutive window sizes. doing that, 8192 and 16384
# produce similar 55 step chroma
# Q: is it worth filtering adjacent components? surely during
# performance, the pitch fluctuates? being strict about this would
# make classification very difficult. maybe it's better to quantise
# to a western scale? or some other scale (e.g. 24 steps)??
def chroma(a, window_size, steps):
    s = Spectrogram(a.signal[:, 0], a.sample_rate, window_size, 2048)
    path = monophonic_maxima_path(s.data)
    notes = notes_from_path(path)
    classes = notes_quantise_pitch_class(notes[:, 2], steps, s.sample_rate,
                                         window_size)
    return pad1d(np.bincount(classes), 0, steps)

def class_from_filename(filename):
    basename = os.path.basename(filename)
    c = basename.split('--', 1)[0]
    return c

def get_training_example(filename):
    import audio

    a = audio.Audio(filename)
    s = Spectrogram(a.signal[:, 0], a.sample_rate, 8192, 2048)
    cls = class_from_filename(filename)
    path = monophonic_path(s.data)
    notes = notes_from_path(path)

    if not len(notes):
        return notes, cls, None

    steps = 12
    pitch_classes = notes_quantise_pitch_class(notes[:, 2], steps, s.sample_rate, s.window_size)
    pc_notes = notes.copy()
    pc_notes[:,2] = pitch_classes
    nklangs = get_nklangs(pc_notes, 4, 2)
    fv = nklangs_to_feature_vector(nklangs, steps)

    return notes, cls, fv

def get_training_data(directory, nfeatures):
    filenames = glob.glob(directory + '/*.pkl')
    examples = np.zeros((len(filenames), nfeatures))
    makams = []
    for i, filename in enumerate(filenames):
        with open(filename, 'rb') as f:
            examples[i, :] = cPickle.load(f)
        makams.append(class_from_filename(filename))

    makams = np.unique(makams, return_inverse=True)[1]

    return examples, makams

def plot_decision_tree(clf):
    import StringIO, pydot 
    dot_data = StringIO.StringIO() 
    sklearn.tree.export_graphviz(clf, out_file=dot_data) 
    graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    graph.write_png('tmp.png') 
    os.system('sxiv tmp.png')

def dt_data(job_data, normalise=True):
    nfeatures = len(job_data.itervalues().next()['fv'])
    examples = np.zeros((len(job_data), nfeatures))
    makams = []
    for i, (key, value) in enumerate(job_data.iteritems()):
        fv = value['fv']
        if normalise:
            fv = fv.astype(float)
            fv /= sum(fv)

        examples[i, :] = fv
        makams.append(class_from_filename(key))

    makams = np.unique(makams, return_inverse=True)[1]

    return examples, makams

def dt_chr_data(data, octave_steps=53):
    examples = np.zeros((len(data), octave_steps))
    makams = []
    for i, (key, value) in enumerate(data.iteritems()):
        notes = value['notes']
        chromagram = notes_to_chromagram(notes, octave_steps, True)
        examples[i, :] = chromagram
        makams.append(class_from_filename(key))

    makams = np.unique(makams, return_inverse=True)[1]

    return examples, makams

def dt_accuracy(examples, makams, training_ratio=.7):

    indices = np.arange(len(makams))
    np.random.shuffle(indices)
    examples = examples[indices,:]
    makams = makams[indices]

    clf = sklearn.tree.DecisionTreeClassifier(min_samples_leaf=20)
    split = int(len(makams) * training_ratio)
    clf.fit(examples[0:split], makams[0:split])
    predicted = clf.predict(examples[split:])
    actual = makams[split:]
    return sum(predicted == actual) / float(len(predicted))

def chr_accuracy(data, training_ratio=.7, octave_steps=53):
    d = [data[k] for k in data.keys()]
    makams = np.unique([class_from_filename(k) for k in data.keys()])
    makams = {k: i for i, k in enumerate(makams)}
    for i, k in enumerate(data.keys()):
        m = class_from_filename(k)
        d[i]['makam'] = makams[m]

    np.random.shuffle(d)
    split = int(len(d) * training_ratio)
    training = d[:split]
    test = d[split:]
    chromagrams = np.zeros((len(makams), octave_steps))

    for item in training:
        c = notes_to_chromagram(item['notes'], octave_steps)
        chromagrams[item['makam'], :] += c

    for i in np.arange(len(makams)):
        s = chromagrams[i, :].sum()
        if s > 0:
            chromagrams[i, :] /= s

    correct = 0
    for item in test:
        c = notes_to_chromagram(item['notes'], octave_steps)
        c = c.astype(float)
        c = c / sum(c)
        predicted = np.argmax((chromagrams * c).sum(1))
        if predicted == item['makam']:
            correct += 1

    return float(correct) / len(test)

def data_by_makam(data):
    by_makam = {}
    for name, value in data.iteritems():
        value = value.copy()
        makam = class_from_filename(name)

        value['name'] = name
        value['chromagram'] = notes_to_chromagram(value['notes'], 53)
            
        if makam in by_makam:
            by_makam[makam].append(value)
        else:
            by_makam[makam] = [value]

    return by_makam

def sum_chromagram(by_makam):
    octave_steps = len(by_makam[by_makam.keys()[0]][0]['chromagram'])
    chromagrams = np.empty((0, octave_steps), dtype=float)
    makams = []
    for i, (makam, items) in enumerate(by_makam.iteritems()):
        if len(items) < 2:
            continue

        c = np.zeros((octave_steps), dtype=float)

        for data in items:
            c += data['chromagram']
        c /= sum(c)
        chromagrams = np.vstack((chromagrams, c))

        makams.append(makam)

    return makams, chromagrams

def plot_notes(notes):
    import matplotlib.pyplot as plt
    plt.step(notes[:,0], notes[:,2], where='post')

# assumes notes have pitchese
def notes_to_chromagram(notes, octave_steps, normalise=False):
    n = notes[:,2]
    n = np.mod(n.astype(int), octave_steps)
    c = np.bincount(n, minlength=octave_steps)
    if normalise:
        c = c / float(sum(c))
    return c

def play_name(name):
    os.system('mpg123 http://andreasjansson.s3.amazonaws.com/makams/mp3/%s.mp3' % name)

def data_prune_tiny(data, n=5):
    dm = data_by_makam(data)
    singles = [k for k, v in dm.iteritems() if len(v) < n]
    return {k: v for k,v in data.iteritems() if class_from_filename(k) not in singles}

def data_to_sequences(data, octave_steps=53):
    sequences = []
    makams = []
    for name, value in data.iteritems():
        notes = np.mod(value['notes'][:,2].astype(int), octave_steps)
        sequences.append(notes)
        makams.append(class_from_filename(name))

    makams = np.unique(makams, return_inverse=True)[1]
    data = zip(makams, sequences)

    return makams, data

def data_to_shapelet_files(data, training_ratio=.7):
    data = data.copy()
    examples = np.array([np.mod(x['notes'][:,2].astype(int), 53) for x in data.values()])
    makams = np.unique([class_from_filename(k) for k in data.keys()],
                       return_inverse=True)[1]

    indices = np.arange(len(makams))
    np.random.shuffle(indices)
    examples = examples[indices,:]
    makams = makams[indices]

    split = int(len(data) * training_ratio)
    training = zip(makams[:split], examples[:split])
    test = zip(makams[split:], examples[split:])

    def write_file(filename, data):
        with open(filename, 'w') as f:
            for makam, example in data:
                f.write('%d ' % makam)
                for note in example:
                    f.write('%d ' % note)
                f.write('\n')

    write_file('shapelet_train.txt', training)
    write_file('shapelet_test.txt', test)
