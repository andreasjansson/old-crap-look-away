import job

def main(job_name, s3_bucket, midi_s3_folder, mp3_s3_folder):
    publisher = job.Job(job_name)

    def paths_to_names(paths):
        return [os.splitext(os.path.basename(p))[0] for p in paths]

    midi_paths = s3.ls(s3_bucket, midi_s3_folder)
    mp3_paths = s3.ls(s3_bucket, midi_s3_folder)
    unprocessed = paths_to_names(midi_paths) - paths_to_names(mp3_paths)

    for name in unprocessed:
        publisher.put_data({
            'bucket': s3_bucket,
            'input': midi_s3_folder + '/' + name,
            'output': mp3_s3_folder + '/' + name,
        })
