from . import *
import string
import pickle
from math import log2
import tables as tb
import numpy as np
import numba as nb
from numba import jit
from scipy.special import kl_div


def remove_symbols(string):

    for symbol in punctuation:
        string = string.replace(symbol, "")
    string = string.lower().split()
    
    return string

def save_dictionary(dictionary, dir):
    
    with open(dir, "wb") as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
        
def load_dictionary(dir):
    with open(dir, 'rb') as f:
        return pickle.load(f)
    
def clip_vocabulary(vocabulary, clip_word_frequency):
    return {key:val for key, val in vocabulary.items() if val > clip_word_frequency}

def load_cooccurrence_matrix():
    return tb.open_file(cooccurrence_matrix_h5_dir, 'r').root.data

def load_distribution_matrix():
    return tb.open_file(distribution_matrix_h5_dir, 'r').root.data

def load_dissimilarity_matrix():
    return tb.open_file(dissimilarity_matrix_h5_dir, 'r').root.data

def kl_divergence(p, q):
    
    if not smoothing:
        diff_sum = 0.
        
        for i in range(len(p)):
            if p[i] != 0 and q[i] != 0:
                diff_sum += p[i] * log2(p[i]/q[i])
            else:
                continue
        
        return diff_sum
    else:
        return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

def l1_norm(p, q):
    return  np.sum(np.absolute(p - q))

def l2_norm(p, q):
    return np.sum([value**2 for value in p-q])

# @jit(nopython=True)
# def kl_divergence_numba(p, q):
#     if not smoothing:
#         diff_sum = 0.
        
#         for i in range(len(p)):
#             if p[i] != 0 and q[i] != 0:
#                 diff_sum += p[i] * log2(p[i]/q[i])
#             else:
#                 continue
        
#         return diff_sum
#     else:
#         return np.sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

@nb.njit
def sum_of_abs(arr):
    sum_ = 0
    for item in arr:
        sum_ += abs(item)
    return sum_