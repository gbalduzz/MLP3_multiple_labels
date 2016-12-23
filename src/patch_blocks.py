import numpy as np
from modules.utils import *
from modules import preprocessing as prp
import h5py
import time

n_train = 278
n_test = 138

#read the data
print("loading training set...")
file = h5py.File("../data/preprocessed/blocks.hdf5", "r")
n_blocks = np.array(file.get("last_done"))[0]

data = np.zeros([n_train+n_test,0])
for i in range(n_blocks):
    idx = threeDindex(i)
    n_keep = keepN(idx)
    block = file.get("block_"+str(i))[:,:n_keep]
    data = np.concatenate((data, block), axis=1)
data = prp.remove_zero_columns(np.array(data), tollerance=1e-5)

print("computing pcr on", data.shape[1], "dims...")
processed = prp.compute_pca(data,-1,scale=True)
print("done!")

out = h5py.File("../data/preprocessed/reduced.hdf5", "w")
out.create_dataset("train_data", data=processed[:n_train,:])
out.create_dataset("test_data", data=processed[n_train:,:])
out.close()



