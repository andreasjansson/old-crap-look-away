import sys
import job
import midi
import s3
import util
import os

def do_work(worker, data):
    out_path = util.tempnam(util.get_extension(data['output']))
    try:
        midi.synthesise(s3.url(data['bucket'], data['input']), out_path, default_program=73) # 73=flute
        s3.upload(data['bucket'], out_path, data['output'])
    finally:
        os.unlink(out_path)

    worker.log('Synthesised %s' % data['input'])
    
if __name__ == '__main__':
    worker = job.Job(sys.argv[1])
    worker.run_worker(do_work)
