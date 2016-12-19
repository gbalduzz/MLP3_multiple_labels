import nibabel
import numpy as np
import os
import modules.preprocessing as prp
#from multiprocessing.pool import ThreadPool

# The brain should be roughly centered by this coords.
bounds = np.array([[19, 154], [24, 186], [8, 152]])

bounds_size = np.prod(bounds[:,1]-bounds[:,0])

def load_directory(dirname, n_blocks):
    """
    :param dirname: relative dir path
    :return: np. array with n_files, n_features dimensions
    """

    path=os.getcwd()+"/"+dirname
    filenames = [name for name in os.listdir(path) if name.split('.')[-1]=='nii']
    n = len(filenames)

    type = filenames[0].split('_')[0]
    assert(type == "train" or type=="test")
    sample_shape = nibabel.load(path+filenames[0]).shape
    four_d = (len(sample_shape) == 4)
    x = []

    #pool = ThreadPool(NUM_THREADS)
    for i in range(n): # work item
        filename = path+"/"+type+"_"+str(i+1)+".nii"
        data=nibabel.load(filename).get_data()
        if four_d: data = data[:,:,:,0]
        x.append(np.ndarray.flatten(prp.blocks(data, n_blocks, bounds)))

    return np.array(x)


def sum_data(filename):
    file = nibabel.load(filename)
    return sum(np.ndarray.flatten(file.get_data()))

def sum_partial_data(filename, boundaries):
    file = nibabel.load(filename)
    data = file.get_data()[boundaries[0][0]:boundaries[0][1],boundaries[1][0]:boundaries[1][1], boundaries[2][0]:boundaries[2][1]]
    return sum(np.ndarray.flatten(data))
