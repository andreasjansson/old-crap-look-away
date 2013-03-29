import urllib
import tempfile
import os

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
