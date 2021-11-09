import numpy as np
from . import *
from .functions import *
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import pandas as pd
import random
import sys


def init_embeddings(dataset, parameters, init_method):
    """
    Initializes word embeddings. 
    INPUTS:     Array of dataset, Array of parameters,
                Str indicating method of initialization, 
                Array of params (or empty array if none required)
    RETURNS:    Dict with [word: np.array], Int of seed"""
    
    def continuous_uniform(parameters):
        """
        Generates random array from continuous uniform distribution 
        with values ranging [0,1).
        INPUTS:     Array of parameters
        RETURNS:    Array of size 'dimensions'"""
        if parameters[2]:
            return (np.random.rand(parameters[0]) - 0.5) * parameters[1]
        else:
            return np.random.rand(parameters[0]) * parameters[1]
    
    def continuous_gaussian(parameters):
        """
        Generates random array from continuous Gaussian distribution.
        INPUTS:     Int indicating dimensions of embeddings
        RETURNS:    Array of size 'dimensions'"""
        
        array = []
        for _ in range(parameters[0]):
            array.append(np.random.normal(parameters[1], parameters[2]))
        
        return np.array(array)
    
    embeddings = {}
    # Generate seed for random methods
    seed = np.random.seed()
    
    for quad in dataset:
        for word in quad:
            if word not in embeddings:
                embeddings[word] = eval(init_method + "(parameters)")
    
    return embeddings, seed

def train(embeddings, 
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
          scale=False,
          method_train="stochastic_gradient_descent",
          method_partition="l2dist",
          print_cost=True):
    
    """Train model. 
    INPUTS:     Vectors, Array of dataset, Array of vocabs, Array of params, 
                Int of epochs,
                Str indicating cost function to use (default: BGD)
    RETURNS:    Vectors (trained)"""
    if partition:
        for _ in trange(partition_epochs):
            embeddings = eval("partitions()." 
                + method_partition 
                + ".train(vocab_dict_by_type, embeddings, params)")
    
    for _ in trange(train_epochs):
        embeddings = eval("cost_functions()." 
             + method_train 
             + ".train(vocab, embeddings, params, dataset, normalization, randomization)")
        if print_cost:
            eval("cost_functions()." 
                    + method_train 
                    + ".calculate_cost(embeddings, dataset)")
    if scale:
        embeddings.update((x, y * scale_factor) for x, y in embeddings.items())
        
    return embeddings
        
def evaluate(embeddings, dataset, vocab):
    """
    Evaluates additive compositionality.
    INPUTS:     Vectors, Array of dataset, Array of vocabs
    RETURNS:    Float of accuracy, Array of incorrect quads"""
    
    correct = 0
    
    incorrect_quads = []; existing_quads = 0; missing_quads = 0
    for quad in tqdm(dataset):
        try:
            v_i = embeddings[quad[0]]
            v_j = embeddings[quad[1]]
            v_k = embeddings[quad[2]]
            
            v_l_prime = v_j - v_i + v_k
            
            closest_word = None; closest_dist = float("inf")
            
            for word in vocab:
                dist = np.linalg.norm(embeddings[word] - v_l_prime)
                if dist < closest_dist:
                    closest_word = word
                    closest_dist = dist
            if closest_word == quad[3]:
                correct += 1
            else:
                incorrect_quads.append(quad)
            existing_quads += 1
        except KeyError:
            missing_quads += 1
    
    accuracy = correct/existing_quads * 100
    print("Accuracy: ", accuracy)
    print("Incorrect quad count: ", len(incorrect_quads))
    print("Missing quad count: ", missing_quads)

    return accuracy, quad

def plot_points(embeddings, vocab, dimensions):
    """Plots learned embeddings in matplotlib. Only works for 2/3 dimensions.
    INPUTS:     Vectors, Array of vocab, 
                Int indicating dimensions
    RETURNS:    None"""
    if dimensions == 2:
        df = pd.DataFrame(columns=["x", "y"])
        
        for index, key in enumerate(vocab.keys()):
            for word in vocab[key]:
                x = np.array(embeddings[word][0])
                y = np.array(embeddings[word][1])
                df = df.append(pd.DataFrame([[x, y]], columns=["x", "y"]))
        fig = px.scatter(df, x="x", y="y")
        fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
        fig.show()
    else:
        df = pd.DataFrame(columns=["x", "y", "z"])

        for index, key in enumerate(vocab.keys()):
            for word in vocab[key]:
                x = np.array(embeddings[word][0])
                y = np.array(embeddings[word][1])
                z = np.array(embeddings[word][2])
                df = df.append(pd.DataFrame([[x, y, z]], columns=["x", "y", "z"]))
        fig = px.scatter_3d(df, x="x", y="y", z="z")
        fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
        fig.update_zaxes(scaleanchor = "x", scaleratio = 1)
        fig.show()

def plot_points_by_type(embeddings, vocab, dimensions):
    """Plots learned embeddings in matplotlib by type. 
    Only works for 2/3 dimensions.
    INPUTS:     Vectors, Array of vocab, 
                Int indicating dimensions
    RETURNS:    None"""

    if dimensions == 2:
        df = pd.DataFrame(columns=["x", "y", "type"])
        
        for index, key in enumerate(vocab.keys()):
            for word in vocab[key]:
                x = np.array(embeddings[word][0])
                y = np.array(embeddings[word][1])
                df = df.append(pd.DataFrame([[x, y, key]], columns=["x", "y", "type"]))
        fig = px.scatter(df, x="x", y="y", color="type")
        fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
        fig.write_html("vis.html")
        fig.show()
    elif dimensions == 3:
        df = pd.DataFrame(columns=["x", "y", "z", "type", "size"])

        for index, key in enumerate(vocab.keys()):
            for word in vocab[key]:
                x = np.array(embeddings[word][0])
                y = np.array(embeddings[word][1])
                z = np.array(embeddings[word][2])
                df = df.append(pd.DataFrame([[x, y, z, key, 10.0]], columns=["x", "y", "z", "type", "size"]))
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="type", size="size")
        fig.write_html("vis.html")
        fig.show()
    
def plot_parallelograms(embeddings, dataset, dimensions, sample_no):
    """Plots a given number of quads and its corresponding parallelogram.
    INPUTS:     Vectors, Array of dataset, Int indicating dimensions, 
                Int indicating number of samples
    RETURNS:    None"""
    
    parallelograms = []
    for ind in random.sample(range(1, len(dataset)), sample_no):
        parallelograms.append(dataset[ind])        

    if dimensions == 2:        
        for parallelogram in parallelograms:
            x = [embeddings[parallelogram[0]][0], 
                 embeddings[parallelogram[1]][0], 
                 embeddings[parallelogram[3]][0], 
                 embeddings[parallelogram[2]][0], 
                 embeddings[parallelogram[0]][0]]
            y = [embeddings[parallelogram[0]][1], 
                 embeddings[parallelogram[1]][1], 
                 embeddings[parallelogram[3]][1], 
                 embeddings[parallelogram[2]][1], 
                 embeddings[parallelogram[0]][1]]
                
            ax.scatter(x, y)
            ax.plot(x, y)
            for index, txt in enumerate(parallelogram):
                ax.annotate(txt, (x[index], y[index]))
        plt.gca().set_aspect('equal', adjustable='box')
        
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for parallelogram in parallelograms:
            x = [embeddings[parallelogram[0]][0], 
                 embeddings[parallelogram[1]][0], 
                 embeddings[parallelogram[3]][0], 
                 embeddings[parallelogram[2]][0], 
                 embeddings[parallelogram[0]][0]]
            y = [embeddings[parallelogram[0]][1], 
                 embeddings[parallelogram[1]][1], 
                 embeddings[parallelogram[3]][1], 
                 embeddings[parallelogram[2]][1], 
                 embeddings[parallelogram[0]][1]]
            z = [embeddings[parallelogram[0]][2], 
                 embeddings[parallelogram[1]][2], 
                 embeddings[parallelogram[3]][2], 
                 embeddings[parallelogram[2]][2], 
                 embeddings[parallelogram[0]][2]]
                    
            ax.scatter(x, y, z)
            ax.plot(x, y, z)
            for index, txt in enumerate(parallelogram):
                ax.text(x[index], y[index], z[index], txt)
        plt.gca().set_aspect('auto', adjustable='box')

    
    plt.show()
        
class cost_functions:
    class batch_gradient_descent:
        def calculate_cost(embeddings, dataset):
            """
            Calculates total cost of vectors. 
            INPUTS:     Vectors, Array of dataset
            RETURNS:    Float value indicating total cost"""
            
            pass
        
        def update_vectors(embeddings, params, quad, normalization):
            """
            Updates word embeddings. 
            INPUTS:     Vectors, Array of params
            RETURNS:    Vectors (updated)"""
            
            pass
            
        def train(vocab, embeddings, param, dataset, normalization):
            """
            Use BGD and train model.
            INPUTS:     Array of vocabs, Vectors, Array of dataset
            RETURNS:    """
            
            pass
        
    class stochastic_gradient_descent:
        def calculate_cost(embeddings, dataset):
            """
            Calculates total cost of vectors. 
            INPUTS:     Vectors, Array of dataset
            RETURNS:    Float value indicating total cost"""

            error = 0
            for quad in dataset:
                error += np.linalg.norm(embeddings[quad[1]] 
                                        - embeddings[quad[0]] 
                                        + embeddings[quad[2]]
                                        - embeddings[quad[3]])
            tqdm.write("Error: " + str(error))
            
            return error
        
        def update_vectors(embeddings, params, quad, normalization):
            """
            Updates word embeddings. 
            INPUTS:     Vectors, Array of params (only learning rate)
            RETURNS:    Vectors (updated)"""
            
            alpha = params[0]
            v_i = embeddings[quad[0]]
            v_j = embeddings[quad[1]]
            v_k = embeddings[quad[2]]
            v_l = embeddings[quad[3]]
            
            if normalization:
                v_i -= (alpha * 2 * (v_i - v_j - v_k + v_l) + beta * 2 * v_i)
                v_j -= (alpha * 2 * (v_j - v_i - v_l + v_k) + beta * 2 * v_j)
                v_k -= (alpha * 2 * (v_k - v_i - v_l + v_j) + beta * 2 * v_k)
                v_l -= (alpha * 2 * (v_l - v_j - v_k + v_i) + beta * 2 * v_l)
            else:
                v_i -= alpha * 2 * (v_i - v_j - v_k + v_l)
                v_j -= alpha * 2 * (v_j - v_i - v_l + v_k)
                v_k -= alpha * 2 * (v_k - v_i - v_l + v_j)
                v_l -= alpha * 2 * (v_l - v_j - v_k + v_i)
            
            embeddings[quad[0]] = v_i
            embeddings[quad[1]] = v_j
            embeddings[quad[2]] = v_k
            embeddings[quad[3]] = v_l

            return embeddings
            
        def train(vocab, embeddings, params, dataset, normalization, 
                  randomization):
            """
            Use SGD and train model.
            INPUTS:     Array of vocabs, Vectors, Array of params, 
                        Array of dataset
            RETURNS:    Vectors"""
            dataset_copy = list(dataset)
            if randomization:
                for x in range(len(dataset)):
                    quad = random.choice(dataset_copy)
                    dataset_copy.remove(quad)
                    embeddings = cost_functions.\
                        stochastic_gradient_descent.\
                            update_vectors(embeddings, params, quad, normalization)
            else:
                for quad in dataset:
                    embeddings = cost_functions.\
                        stochastic_gradient_descent.\
                            update_vectors(embeddings, params, quad, normalization)
            return embeddings
        
class partitions:
    class l2dist:
        def train(vocab_dict_by_type, embeddings, params):
            """
            Updates vector position by type.
            INPUT:      Dict of vocabs by type, Vectors, Array of params
            RETURNS:    Vectors"""
            
            for word_1 in vocab_dict_by_type:
                word_type = vocab_dict_by_type[word_1]
                for word_2 in vocab_dict_by_type:
                    if word_1 != word_2:
                        
                        v_i = embeddings[word_1]
                        v_j = embeddings[word_2]
                        gamma = params[1]
                        
                        if word_type == vocab_dict_by_type[word_2]:
                            v_i -= gamma * 2 * (v_i - v_j)
                        else:
                            v_i -= gamma * -1 * 2 * (v_i - v_j)
                        
                        embeddings[word_1] = v_i
                    
            return embeddings