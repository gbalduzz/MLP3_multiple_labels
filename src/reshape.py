from modules import file_IO, preprocessing
import numpy as np
import h5py

# Number of kept dimensions per block.
block =np.array([7,7,7])

n_blocks = np.prod(block)
#Load the dataset
print("loading training set...")
data = file_IO.load_directory("../data/set_train/", block)
n_train = data.shape[0]

print("loading testing set...")
data = np.concatenate((data,
                      file_IO.load_directory("../data/set_test/", block)
                     ), axis = 0)

print("applying blocked pca...")
f = h5py.File("../data/preprocessed/reshaped.hdf5", "w")
f.create_dataset("data", data=data)
f.create_dataset("n_train", data=n_train)
f.create_dataset("blocks", data=block)



