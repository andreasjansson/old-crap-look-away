import spectrogram
import cPickle
import os
import time
import sys
import glob

if __name__ == '__main__':
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    total_time = 0
    input_filenames = sorted(glob.glob(input_directory + '/*.mp3'))
    for i, filename in enumerate(input_filenames):
        start_time = time.time()
        c, fv = spectrogram.get_training_example(filename)
        name = os.path.splitext(os.path.basename(filename))[0]
        output_filename = '%s/%s.pkl' % (output_directory, name)

        with open(output_filename, 'wb') as f:
            cPickle.dump(fv, f)

        total_time += time.time() - start_time

        print '%d/%d: %s (avg time: %.1f seconds)' % (
            i + 1, len(input_filenames), name, total_time / (i + 1))
