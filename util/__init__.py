#? Data directory
dataset_dir = "data/analogy/google_analogy_dataset.txt"
# dataset_dir = "data/analogy/bats_3.0_combined.txt"
similarity_dataset_dir = "data/similarity/SimVerb-3500/SimVerb-3500.txt"

#? Parameters
train_epochs = 300
partition_epochs = 500
dimensions = 2
init_method = "continuous_uniform"
reverse = False
partition = False
normalization = False
randomization = False
scale_factor = 10000
scaling = False
train_type = "tsne_distribution"

#? Parameters for initialization
uniform_sigma = 1.0
gaussian_mu = 0
centering = True
gaussian_sigma = 1.0

#? Parameters for BGD and SGD
alpha = 0.001
beta = 0.0001
gamma = 0.0001

#? Parameters for plotting
colors = ["#FF0000", 
          "#00FFFF", 
          "#C0C0C0", 
          "#0000FF", 
          "#808080", 
          "#00008B", 
          "#000000", 
          "#ADD8E6", 
          "#FFA500", 
          "#800080", 
          "#A52A2A", 
          "#FFFF00", 
          "#800000", 
          "#00FF00", 
          "#008000", 
          "#FF00FF", 
          "#808000", 
          "#FFC0CB", 
          "#7FFD4"]