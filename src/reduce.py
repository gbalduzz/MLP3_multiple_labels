from modules import file_IO, preprocessing
import numpy as np
import h5py

# Number of kept dimensions per block.
k1 = 520
block =np.array([7,7,7])
# Number of final dimensions
k2 = 100000
f = h5py.File("../data/preprocessed/reduced.hdf5", "w")

n_blocks = np.prod(block)
def load_component(comp_name):
    print("loading training set...")
    data = file_IO.load_directory("../data/set_train/"+comp_name+"/", block)
    n_train = data.shape[0]

    print("loading testing set...")
    data = np.concatenate((data,
                           file_IO.load_directory("../data/set_test/" + comp_name + "/", block)
                          ), axis = 0)

    print("applying blocked pca...")
    data = preprocessing.blocked_pca(data, n_blocks, k1)
    print("applying second pca...")
    data = preprocessing.compute_pca(data, k2, scale=True)
    return data, n_train


data, n_train = load_component("")

print("Saving to disk.")
f.create_dataset("train_data", data=data[:n_train, :])
f.create_dataset("test_data",  data=data[n_train:, :])

