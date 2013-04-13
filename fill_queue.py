import job
import sys
import s3
import os

def main(job_name, s3_bucket, midi_s3_folder, mp3_s3_folder):
    publisher = job.Job(job_name)

    def paths_to_names(paths):
        if paths is None:
            return set([])
        return set([os.path.splitext(os.path.basename(p))[0] for p in paths])

    midi_paths = s3.ls(s3_bucket, midi_s3_folder)
    mp3_paths = []#s3.ls(s3_bucket, mp3_s3_folder)
    unprocessed = paths_to_names(midi_paths) - paths_to_names(mp3_paths)

    #publisher.clear()

    for name in unprocessed:
        data = {
            'bucket': s3_bucket,
            'input': midi_s3_folder + '/' + name + '.mid',
            'output': mp3_s3_folder + '/' + name + '.mp3',
        }
        print 'Adding to queue: %s' % name
        publisher.put_data(data)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
