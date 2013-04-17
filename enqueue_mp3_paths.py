import job
import sys
import s3
import os

def main(job_name):
    publisher = job.Job(job_name)

    mp3_paths = s3.ls('andreasjansson', 'makams/mp3', r'\.mp3')

    for path in mp3_paths:
        path = s3.url('andreasjansson', path)
        data = {
            'path': path
        }
        print 'Adding to queue: %s' % path
        publisher.put_data(data)

if __name__ == '__main__':
    main(sys.argv[1])
