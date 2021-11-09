import string


allowed_punctuation = "'"
punctuation = string.punctuation.replace(allowed_punctuation, "")
lowercase = False
clip_word_frequency = 10
window_size = 4
docs_to_look_at = 1000

vocabulary_dir = "data/document/files/vocabulary_small.pkl"
vocabulary_sample_dir = "data/document/files/sample/vocabulary_sample.pkl"

cooccurrence_matrix_h5_dir = "data/document/files/cooccurrence_matrix_small.h5"
cooccurrence_matrix_h5_sample_dir = "data/document/files/sample/cooccurrence_matrix_sample.h5"

distribution_matrix_h5_dir = "data/document/files/distribution_matrix_small.h5"
distribution_matrix_h5_sample_dir = "data/document/files/sample/distribution_matrix_sample.h5"

dissimilarity_matrix_h5_dir = "data/document/files/dissimilarity_matrix_small.h5"
dissimilarity_matrix_h5_sample_dir = "data/document/files/sample/dissimilarity_matrix_sample.h5"

sample_doc = "the brown fox jumped over the yellow fence."
not_sample = True
smoothing = True