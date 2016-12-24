import nibabel
import numpy as np
import os
import modules.preprocessing as prp

def crop(x, n_blocks):
    """"
    cuts the 3D x array so that it is divisible by n_blocks, should roughly center the brain.
    """
    # enlarge this until the division is exact
    initial_limit = np.array([[19,154], [24,186], [8,152]])

    remainder = (n_blocks - ((initial_limit[:, 1] - initial_limit[:, 0]) % n_blocks)) % n_blocks
    limit = initial_limit
    limit[:, 0] = initial_limit[:, 0] - remainder / 2
    limit[:, 1] = initial_limit[:, 1] + remainder / 2 + (remainder % 2)

    return x[limit[0,0]:limit[0,1], limit[1,0]:limit[1,1], limit[2,0]:limit[2,1]]


def load_directory(dirname):
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
    #n_features = np.prod(sample_shape)
    n = 5# for DEBUG purposes
    x = np.zeros([n,sample_shape[0],sample_shape[1],sample_shape[2],1])

    for i in range(n): # n
        filename = path+"/"+type+"_"+str(i+1)+".nii"
        x[i]= nibabel.load(filename).get_data()

    return x

