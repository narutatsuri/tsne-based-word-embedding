from . import *
from .functions import *
import numpy as np
from scipy.sparse import csr_matrix
import tables as tb
from tqdm import tqdm
import sys
from datasets import load_dataset
import os.path


def load_vocabulary():
    
    vocabulary = {}
    
    if not_sample and not os.path.isfile(vocabulary_dir):
        wiki = load_dataset("wikipedia", "20200501.en", split='train')

        for doc_index in tqdm(range(docs_to_look_at)):
            parsed_doc = remove_symbols(wiki[doc_index]["text"])
            for word in parsed_doc:
                if word in vocabulary:
                    vocabulary[word] += 1
                else:
                    vocabulary[word] = 1

        save_dictionary(vocabulary, vocabulary_dir)
        
    elif not os.path.isfile(vocabulary_sample_dir):
        parsed_doc = remove_symbols(sample_doc)
        for word in parsed_doc:
            if word in vocabulary:
                vocabulary[word] += 1
            else:
                vocabulary[word] = 1

        save_dictionary(vocabulary, vocabulary_sample_dir)
        
def generate_matrix():
    
    if not_sample and not os.path.isfile(cooccurrence_matrix_h5_dir):
        wiki = load_dataset("wikipedia", "20200501.en", split='train')
        vocabulary = load_dictionary(vocabulary_dir)
        # Strip bottom words
        vocabulary = clip_vocabulary(vocabulary, clip_word_frequency)
        vocabulary_row = list(vocabulary.keys())
        vocabulary_size = len(vocabulary)

        cooccurrence_matrix = csr_matrix((vocabulary_size, vocabulary_size), dtype=np.int8)
        
        f = tb.open_file(cooccurrence_matrix_h5_dir, 'w')
        filters = tb.Filters(complevel=5, complib='blosc')
        cooccurrence_matrix = f.create_carray(f.root, 
                                            "data", 
                                            tb.Float16Atom(), 
                                            shape=cooccurrence_matrix.shape, 
                                            filters=filters)

        for doc_index in tqdm(range(docs_to_look_at)):
            
            parsed_doc = remove_symbols(wiki[doc_index]["text"])
            
            for word_index, word in enumerate(parsed_doc):
                window_words = parsed_doc[max(word_index-window_size, 0):\
                    min(word_index+window_size+1, len(parsed_doc))]
                window_words.remove(word)
                
                for window_word in window_words:
                    try:
                        index_i = vocabulary_row.index(word)
                        index_j = vocabulary_row.index(window_word)
                        
                        cooccurrence_matrix[index_i, index_j] += 1
                    except ValueError:
                        continue
        f.close()                
    
    elif not os.path.isfile(cooccurrence_matrix_h5_sample_dir):
        vocabulary = load_dictionary(vocabulary_sample_dir)
        vocabulary_row = list(vocabulary.keys())
        vocabulary_size = len(vocabulary)

        cooccurrence_matrix = csr_matrix((vocabulary_size, vocabulary_size), dtype=np.float16)
        
        f = tb.open_file(cooccurrence_matrix_h5_sample_dir, 'w')
        filters = tb.Filters(complevel=5, complib='blosc')
        cooccurrence_matrix = f.create_carray(f.root, 
                                            "data", 
                                            tb.Float16Atom(), 
                                            shape=cooccurrence_matrix.shape, 
                                            filters=filters)

            
        parsed_doc = remove_symbols(sample_doc)

        for word_index, word in enumerate(parsed_doc):
            window_words = parsed_doc[max(word_index-window_size, 0):\
                min(word_index+window_size+1, len(parsed_doc))]
            window_words.remove(word)

            for window_word in window_words:
                try:
                    index_i = vocabulary_row.index(word)
                    index_j = vocabulary_row.index(window_word)
                    
                    cooccurrence_matrix[index_i, index_j] += 1
                except ValueError:
                    continue
        f.close()

def generate_distribution():
    
    if not_sample and not os.path.isfile(distribution_matrix_h5_dir):
        vocabulary = load_dictionary(vocabulary_dir)
        cooccurrence_matrix_h5 = tb.open_file(cooccurrence_matrix_h5_dir, 'r+')
        
        cooccurrence_matrix = cooccurrence_matrix_h5.root.data
        row_size = len(cooccurrence_matrix[0,:])
        
        distribution_matrix = csr_matrix((row_size, row_size), dtype=np.float16)
        f = tb.open_file(distribution_matrix_h5_dir, 'w')
        filters = tb.Filters(complevel=5, complib='blosc')
        distribution_matrix = f.create_carray(f.root, 
                                            "data", 
                                            tb.Float16Atom(), 
                                            shape=distribution_matrix.shape, 
                                            filters=filters)
        
    elif not os.path.isfile(distribution_matrix_h5_sample_dir):
        vocabulary = load_dictionary(vocabulary_sample_dir)
        cooccurrence_matrix_h5 = tb.open_file(cooccurrence_matrix_h5_sample_dir, 'r+')
        
        cooccurrence_matrix = cooccurrence_matrix_h5.root.data
        row_size = len(cooccurrence_matrix[0,:])
        
        distribution_matrix = csr_matrix((row_size, row_size), dtype=np.float16)
        f = tb.open_file(distribution_matrix_h5_sample_dir, 'w')
        filters = tb.Filters(complevel=5, complib='blosc')
        distribution_matrix = f.create_carray(f.root, 
                                            "data", 
                                            tb.Float16Atom(), 
                                            shape=distribution_matrix.shape, 
                                            filters=filters)
    else:
        return

    for index_i in tqdm(range(row_size)):
        if not smoothing:
            row_i = cooccurrence_matrix[index_i, :]
        else:
            row_i = cooccurrence_matrix[index_i, :] + 1
        row_i = row_i/sum(row_i)
        distribution_matrix[index_i, :] = row_i
    f.close()

def generate_dissimilarlity():
    if not_sample:
        vocabulary = load_dictionary(vocabulary_dir)
        distribution_matrix_h5 = tb.open_file(distribution_matrix_h5_dir, 'r')
    else:
        vocabulary = load_dictionary(vocabulary_sample_dir)
        distribution_matrix_h5 = tb.open_file(distribution_matrix_h5_sample_dir, 'r')
        
    distribution_matrix = distribution_matrix_h5.root.data
    row_size = len(distribution_matrix[0,:])
    
    if not_sample:
        dissimilarity_matrix = csr_matrix((row_size, row_size), dtype=np.float16)
        f = tb.open_file(dissimilarity_matrix_h5_dir, 'w')
    else:
        dissimilarity_matrix = csr_matrix((row_size, row_size), dtype=np.float16)
        f = tb.open_file(dissimilarity_matrix_h5_sample_dir, 'w')

    filters = tb.Filters(complevel=5, complib='blosc')
    dissimilarity_matrix = f.create_carray(f.root, 
                                        "data", 
                                        tb.Float16Atom(), 
                                        shape=dissimilarity_matrix.shape, 
                                        filters=filters)
            
    for index_i in tqdm(range(row_size)):
        for index_j in tqdm(range(index_i, row_size), leave=False):
            if index_i != index_j:
                row_i = distribution_matrix[index_i, :]
                row_j = distribution_matrix[index_j, :]
                
                dissimilarity_matrix[index_i, index_j] = (kl_divergence(row_i, row_j) + kl_divergence(row_j, row_i))/2.
                dissimilarity_matrix[index_j, index_i] = (kl_divergence(row_i, row_j) + kl_divergence(row_j, row_i))/2.
    
    if not not_sample:
        print(dissimilarity_matrix[:,:])
        
    f.close()