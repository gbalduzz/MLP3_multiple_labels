from modules import file_IO, preprocessing
import numpy as np
import h5py

# Size of the data block to be averaged.
n_blocks = np.array([7,7,7])
n_bins = 50

f = h5py.File("preprocessed/reduced.hdf5", "w")

def load_component(comp_name):
    """
    return: 1) array of shape n_train+n_test, comp_size
            2) n_train
    """
    print("loading training set...")
    data = file_IO.load_directory("../data/set_train/"+comp_name+"/", n_blocks, n_bins)
    n_train = data.shape[0]
    print("loading test set...")
    data = np.concatenate((data,
                           file_IO.load_directory("../data/set_test/" + comp_name + "/", n_blocks, n_bins)
                          ), axis = 0)
    return preprocessing.remove_zero_columns(data, n_train), n_train
    #return data, n_train

data, n_train = load_component("")
#data = np.concatenate((data1, data2, data3), axis=1)

f.create_dataset("train_data", data=data[:n_train, :])
f.create_dataset("test_data",  data=data[n_train:, :])
print("n train dataset: ", n_train)

f.close()
