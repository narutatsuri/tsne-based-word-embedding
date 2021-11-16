import numpy as np
from util import *
from util.modules import *
from util.functions import *
from data.document.util import *
from data.document.util.functions import *
from data.document.util.modules import *
from sklearn.manifold import TSNE
from sklearn.manifold import MDS


def train_parallelograms():
    # Load dataset and embeddings
    full_dataset = load_dataset(reverse=reverse)
    dataset = load_partial_dataset()
    print(len(dataset))
    vocab = load_vocabulary(dataset)
    vocab_by_type = load_vocabulary_by_type()
    vocab_dict_by_type = load_vocabulary_dict_by_type()
    
    embeddings, seed = init_embeddings(dataset, 
                                       [dimensions, 
                                        uniform_sigma,
                                        centering], 
                                       init_method)

    # Define cost functions and parameters to use
    method = "stochastic_gradient_descent"
    params = [alpha, gamma]
    
    # Train
    embeddings = train(embeddings, 
                       dataset,
                       vocab, 
                       vocab_dict_by_type,
                       params, 
                       train_epochs,
                       partition_epochs, 
                       partition,
                       normalization,
                       randomization,
                       scale_factor,
                       scale=scaling,
                       print_cost=False)
    
    # Get accuracy
    accuracy, quad = evaluate(embeddings, full_dataset, vocab)
    
    # Plotting
    dataset_type = load_dataset_by_type(": family")
    # plot_points(embeddings, vocab, dimensions)
    plot_points_by_type(embeddings, vocab_by_type, dimensions)
    # plot_parallelograms(embeddings, dataset_type, dimensions, 10)

    # Save results
    save_results(reverse, dimensions, init_method, seed, embeddings)
    
def train_tsne_dissimilarity(vocabulary, dimensions):
    dissimilarity_matrix = np.array(load_dissimilarity_matrix())
    
    t_sne = TSNE(metric="precomputed", verbose=1, n_components=dimensions)
    embeddings = t_sne.fit_transform(dissimilarity_matrix)
    
    return embeddings
    
def train_tsne_distribution(vocabulary, dimensions):
    distribution_matrix = np.array(load_distribution_matrix())
    
    t_sne = modified_TSNE(verbose=1, n_components=dimensions, method="normal_kl")
    embeddings = t_sne.fit_transform(distribution_matrix)
    
def mds(vocabulary, dimensions):
    dissimilarity_matrix = np.array(load_dissimilarity_matrix())

    mds = MDS(n_components=2)
    embeddings = mds.fit_transform(dissimilarity_matrix)
    
    return embeddings