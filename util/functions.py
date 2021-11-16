from . import *
# import pyximport
# pyximport.install()
# from ._barnes_hut_tsne import *
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
    
# def _kl_divergence_bh(params,
#                       P,
#                       degrees_of_freedom,
#                       n_samples,
#                       n_components,
#                       angle=0.5,
#                       skip_num_points=0,
#                       verbose=False,
#                       compute_error=True,
#                       num_threads=1):
    
#     """t-SNE objective function: KL divergence of p_ijs and q_ijs.

#     Uses Barnes-Hut tree methods to calculate the gradient that
#     runs in O(NlogN) instead of O(N^2).

#     Parameters
#     ----------
#     params : ndarray of shape (n_params,)
#         Unraveled embedding.

#     P : sparse matrix of shape (n_samples, n_sample)
#         Sparse approximate joint probability matrix, computed only for the
#         k nearest-neighbors and symmetrized. Matrix should be of CSR format.

#     degrees_of_freedom : int
#         Degrees of freedom of the Student's-t distribution.

#     n_samples : int
#         Number of samples.

#     n_components : int
#         Dimension of the embedded space.

#     angle : float, default=0.5
#         This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
#         'angle' is the angular size (referred to as theta in [3]) of a distant
#         node as measured from a point. If this size is below 'angle' then it is
#         used as a summary node of all points contained within it.
#         This method is not very sensitive to changes in this parameter
#         in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
#         computation time and angle greater 0.8 has quickly increasing error.

#     skip_num_points : int, default=0
#         This does not compute the gradient for points with indices below
#         `skip_num_points`. This is useful when computing transforms of new
#         data where you'd like to keep the old data fixed.

#     verbose : int, default=False
#         Verbosity level.

#     compute_error: bool, default=True
#         If False, the kl_divergence is not computed and returns NaN.

#     num_threads : int, default=1
#         Number of threads used to compute the gradient. This is set here to
#         avoid calling _openmp_effective_n_threads for each gradient step.

#     Returns
#     -------
#     kl_divergence : float
#         Kullback-Leibler divergence of p_ij and q_ij.

#     grad : ndarray of shape (n_params,)
#         Unraveled gradient of the Kullback-Leibler divergence with respect to
#         the embedding.
#     """
#     params = params.astype(np.float32, copy=False)
#     X_embedded = params.reshape(n_samples, n_components)

#     val_P = P.data.astype(np.float32, copy=False)
#     neighbors = P.indices.astype(np.int64, copy=False)
#     indptr = P.indptr.astype(np.int64, copy=False)

#     grad = np.zeros(X_embedded.shape, dtype=np.float32)
#     error = _barnes_hut_tsne.gradient(
#         val_P,
#         X_embedded,
#         neighbors,
#         indptr,
#         grad,
#         angle,
#         n_components,
#         verbose,
#         dof=degrees_of_freedom,
#         compute_error=compute_error,
#         num_threads=num_threads)
    
#     c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
#     grad = grad.ravel()
#     grad *= c

#     return error, grad

def _kl_divergence(
    params,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    skip_num_points=0,
    compute_error=True,
):
    """t-SNE objective function: gradient of the KL divergence
    of p_ijs and q_ijs and the absolute error.

    Parameters
    ----------
    params : ndarray of shape (n_params,)
        Unraveled embedding.

    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.

    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    skip_num_points : int, default=0
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.

    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.

    grad : ndarray of shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS

    # Objective: C (Kullback-Leibler divergence of P and Q)
    if compute_error:
        kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan

    # Gradient: dC/dY
    # pdist always returns double precision distances. Thus we need to take
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order="K"), X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad
