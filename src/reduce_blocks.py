from modules import file_IO, preprocessing
import numpy as np
import h5py

# Number of kept dimensions per block.
k1 = 520
blocks = 7**3

#Load the dataset
print("loading data set...")
file = h5py.File("../data/preprocessed/reshaped.hdf5", "r")
data = np.array(file.get("data"))

print("applying blocked pca...")
f = h5py.File("../data/preprocessed/blocks.hdf5")
preprocessing.blocked_pca(data, blocks, k1, file)



