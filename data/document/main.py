import numpy as np
from util import *
from util.modules import *
from util.functions import *
import sys

load_vocabulary()
print("Loaded vocabulary")

generate_matrix()
print("Generated co-occurrence matrix")

generate_distribution()
print("Generated distribution")

generate_dissimilarlity()
print("Generated dissimilarity")