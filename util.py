import urllib
import tempfile
import os
import matplotlib.pyplot as plt

def get_extension(path):
    split = os.path.splitext(path)
    if len(split) <= 1:
        return ''
    return split[1]

def tempnam(extension):
    f = tempfile.NamedTemporaryFile(suffix=extension, delete=False)
    f.close()
    return f.name

def make_local(path):
    if path.startswith('http://') or path.startswith('https://'):
        return download(path)
    else:
        return path

def download(path):
    temp = tempnam(get_extension(path))
    urllib.urlretrieve(path, temp)
    return temp

def multiplot(plot_functions):
    fig, axes = plt.subplots(len(plot_functions))
    for ax, function in zip(axes, plot_functions):
        function(ax)
    fig.show()

def wav_to_mp3(wav_filename):
    if not os.path.exists(wav_file):
        raise Exception('%s does not exist')

    mp3_filename = tempnam('.mp3')

    lame_output = subprocess.check_output(
        ['lame', '-b128', wav_filename, mp3_filename], shell = False,
        stderr=subprocess.STDOUT)

    if os.path.exists(mp3_filename):
        return mp3_filename

    raise Exception('Failed to convert %s to %s: %s' % (
        wav_filename, mp3_filename, lame_output))
