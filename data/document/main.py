import numpy as np
from util import *
from util.modules import *
from util.functions import *
import sys


if large_mode:
    load_vocabulary()
    vocabulary = load_dictionary(vocabulary_dir)
    print("Loaded vocabulary, vocab size: ", len(vocabulary))

    generate_matrix()
    print("Generated co-occurrence matrix")

    generate_distribution()
    print("Generated distribution")

    generate_dissimilarlity()
    print("Generated dissimilarity")

else:
    vocabulary = load_vocabulary_ram()
    print("Loaded vocabulary, vocab size: ", len(vocabulary))

    cooccurrence_matrix = generate_matrix_ram(vocabulary)
    print("Generated co-occurrence matrix")
    np.save("cooccurrence_matrix.npy", cooccurrence_matrix)

    distribution_matrix = generate_distribution_ram(vocabulary, cooccurrence_matrix)
    print("Generated distribution")
    np.save("distribution_matrix.npy", distribution_matrix)

    dissimilarity_matrix = generate_dissimilarlity_ram(vocabulary, distribution_matrix)
    print("Generated dissimilarity")
    np.save("dissimilarity_matrix.npy", dissimilarity_matrix)
