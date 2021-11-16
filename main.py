import numpy as np
from util.methods import *
import sys
from data.document.util.functions import *


if train_type == "tsne_dissimilarity":
    vocabulary = load_dictionary(vocabulary_dir)
    embeddings = train_tsne_dissimilarity(vocabulary, dimensions)

    plot_points(embeddings, dimensions)

elif train_type == "tsne_distribution":
    vocabulary = load_dictionary(vocabulary_dir)
    embeddings = train_tsne_distribution(vocabulary, dimensions)
    
    plot_points(embeddings, dimensions)

elif train_type == "parallelograms":
    train_parallelograms()