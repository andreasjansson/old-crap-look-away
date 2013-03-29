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

def download(path):
    temp = tempnam(get_extension(path))
    urllib.urlretrieve(path, temp)
    return temp

def multiplot(plot_functions):
    fig, axes = plt.subplots(len(plot_functions))
    for ax, function in zip(axes, plot_functions):
        function(ax)
    fig.show()
