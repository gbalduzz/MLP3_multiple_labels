from modules import file_IO, preprocessing
import numpy as np
import h5py

# Number of kept dimensions.
k = 100000

f = h5py.File("../data/preprocessed/reduced.hdf5", "w")

def load_component(comp_name):
    print("loading training set...")
    train = file_IO.load_directory("../data/set_train/"+comp_name+"/")

    train, U, S = preprocessing.compute_pca(train, k)

    print("loading test set...")
    test = file_IO.load_directory("../data/set_test/" + comp_name + "/")
    test = preprocessing.apply_pca(test, U, S)

    return train, test


train, test = load_component("")

print("Saving to disk.")
f.create_dataset("train_data", data = train)
f.create_dataset("test_data",  data = test)

