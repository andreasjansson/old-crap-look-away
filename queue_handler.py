import job
import midi
import s3
import util

def do_work(worker, data):
    with util.tempnam(util.get_extension(data['output'])) as out_path:
        midi.synthesise(data['input'], out_path, default_program=73) # 73=flute
        s3.upload(data['bucket'], out_path, data['output'])

    worker.log('Synthesised %s' % data['path'])
    
if __name__ == '__main__':
    worker = job.Job(sys.argv[1])
    worker.run(do_work)
