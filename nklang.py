import numpy as np
import math
import scipy
import matplotlib.pyplot as plt

def get_nklangs(chromagram_data, n = 2):
    threshold = 12 - n
    chromagram_data = np.argsort(chromagram_data, 0)
    indices = np.where(chromagram_data < threshold)
    nklangs = indices[0][np.argsort(indices[1])]
    nklangs.shape = (n, len(nklangs) / n)
    return nklangs
