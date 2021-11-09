from . import *
import itertools
import random


def load_dataset(reverse):
    """
    Loads dataset. Reverse positions of pairs if reverser is True.
    INPUTS:     Bool of reverse
    RETURNS:    Array of strs with 4 words [i, j, k, l],
                where v_j - v_i + v_k = v_l"""
    dataset = []
    
    with open(dataset_dir, "r") as f:
        for line in f.readlines():
            # Skip if line is label
            if line[0] == ":":
                continue
            if not reverse:
                dataset.append(line.split())
            else:
                split_line = line.split()
                reverse = [split_line[1], 
                           split_line[0], 
                           split_line[3], 
                           split_line[2]]
                dataset.append(reverse)
                
    return dataset

def load_partial_dataset():
    dataset = []
    
    with open(dataset_dir, "r") as f:
        quads = []
        for line in f.readlines():
            # Skip if line is label
            if line[0] == ":":
                if len(quads) != 0:
                    selected_quads = random.sample(quads, int(len(quads)))
                    dataset += selected_quads
                    quads = []
            else:
                quads.append(line.split())

    return dataset

def load_vocabulary(dataset):
    """
    Load vocabulary. 
    INPUTS:     Array of dataset
    RETURNS:    Array of all words"""
        
    return set(itertools.chain.from_iterable(dataset))

def load_vocabulary_by_type():
    """
    Load vocabulary by type. 
    INPUTS:     None
    RETURNS:    Dict of array of all words and their type"""
    dataset = {}
    
    with open(dataset_dir, "r") as f:
        current_key = None; current_set = set()
        for line in f.readlines():
            if line[0] == ":":
                dataset[current_key] = current_set
                current_set = set()
                current_key = line.replace(":", "")
                current_key = line.replace("\n", "")
                continue
            split_line = line.split()
            current_set.update([word for word in split_line])                

    return dataset

def load_vocabulary_dict_by_type():
    """
    Load vocabulary by type where each word is a key. 
    INPUTS:     None
    RETURNS:    Dict of array of all words and their type"""
    dataset = {}
    
    with open(dataset_dir, "r") as f:
        current_key = None
        for line in f.readlines():
            if line[0] == ":":
                current_key = line.replace(":", "")
                current_key = line.replace("\n", "")
                continue
            split_line = line.split()
            for word in split_line:
                if word not in dataset:
                    dataset[word] = current_key

    return dataset
        

def load_dataset_by_type(quad_type):
    """
    Loads dataset by quadruple type. 
    INPUTS:     None
    RETURNS:    Array of strs with 4 words [i, j, k, l],
                where v_j - v_i + v_k = v_l"""
    dataset = []
    
    with open(dataset_dir, "r") as f:
        lines = f.readlines()
        
        for index, line in enumerate(lines):
            if quad_type in line:
                start = index + 1
        
        for index, line in enumerate(lines[start:]):
            if ":" in line:
                end = start + index
                break
        
        for line in lines[start:end]:
            dataset.append(line.split())

    return dataset
    

def column(matrix, i):
    """
    Gets column of matrix. 
    INPUTS:     Array, Int of column to look at
    RETURNS:    Array of the column"""
    
    return [row[i] for row in matrix]

def save_results(reverse, dimensions, init_method, seed, embeddings):
    """
    Saves results of current iteration into a text file.
    INPUTS:     Bool of reverse, Int of dimensions, Str of initialization 
                method, Int of seed, Vectors
    RETURNS:    None
    """
    with open("results.txt", "a") as f:
        f.write(str(reverse) + ":" + str(dimensions) + ":" 
                + init_method + ":" 
                + str(seed) + ":" 
                + str(embeddings))
    
    print("Saved to " + "results.txt")