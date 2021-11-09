from . import *
import string
import pickle
from math import log2


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
