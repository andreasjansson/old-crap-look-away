import os
import util
import subprocess

def synthesise(input_path, output_path, default_program=None):

    if util.get_extension(input_path) != '.mid':
        raise SynthesiseException('%s is not a .mid file' % input_path)

    output_extension = util.get_extension(output_path)
    if output_extension not in ['.wav', '.mp3']:
        raise SynthesiseException('Cannot synthesise to %s' % output_extension)

    midi_filename = util.make_local(input_path)
    if output_extension == '.wav':
        wav_filename = output_path
    else:
        wav_filename = util.tempnam('.wav')

    options = ['timidity', '-Ow', '-o' + wav_filename, midi_filename]
    if default_program is not None:
        options = options + ['--default-program', str(default_program)]

    timidity_output = subprocess.check_output(
        options, shell = False, stderr=subprocess.STDOUT)

    if midi_filename != input_path:
        os.unlink(midi_filename)

    if not os.path.exists(wav_filename):
        raise SynthesiseException('Failed to synthesise %s: %s', wav_filename, timidity_output)

    if output_extension == '.wav':
        return wav_filename

    try:
        mp3_filename = util.wav_to_mp3(wav_filename, output_path)
    except Exception, e:
        raise SynthesiseException('Failed to convert wav to mp3: ' + str(e))
    finally:
        os.unlink(wav_filename)

    return mp3_filename

class SynthesiseException(Exception):
    pass
