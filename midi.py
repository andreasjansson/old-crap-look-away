import os
import util

def synthesise(input_path, output_path, default_program=None):

    if util.get_extension(input_path) != '.mid':
        raise SynthesiseException('%s is not a .mid file' % input_path)

    output_extension = util.get_extension(output_path)
    if output_extension not in ['.wav', '.mp3']:
        raise SynthesiseException('Cannot synthesise to %s' % output_extension)

    additional_options = []
    if default_program is not None:
        additional_options.append('--default-program', default_program)

    midi_filename = util.make_local(input_path)
    wav_filename = util.tempnam('.wav')
    timidity_output = subprocess.check_output(
        ['timidity', '-Ow', '-o' + wav_filename, midi_filename] +
        additional_options, shell = False,
        stderr=subprocess.STDOUT)

    if midi_filename != input_path:
        os.path.unlink(midi_filename)

    if not os.path.exists(wav_filename):
        raise SynthesiseException('Failed to synthesise %s: %s', wav_filename, timidity_output)

    if output_extension == '.wav':
        return wav_filename

    try:
        mp3_filename = util.wav_to_mp3(wav_filename)
    except Exception, e:
        raise SynthesiseException('Failed to convert wav to mp3: ' + str(e))
    finally:
        os.path.unlink(wav_filename)

    return mp3_filename

class SynthesiseException(Exception):
    pass
