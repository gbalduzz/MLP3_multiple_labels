import file_IO, preprocessing
import numpy as np

# Mimicss the reduce.py script but with choosable parameters.
def load_data(n_blocks =9, n_bins=65, set="train"):
    block_array = np.array([n_blocks,n_blocks,n_blocks])
    data = file_IO.load_directory("../data/set_"+set+"/", block_array, n_bins)
    #return preprocessing.remove_zero_columns(data, n_train), n_train
    return data

