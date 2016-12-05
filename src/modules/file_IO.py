import nibabel
import numpy as np
import os
import modules.preprocessing as prp
#from multiprocessing.pool import ThreadPool



def load_directory(dirname, prep_function):
    """
    :param dirname: relative dir path
    :return: np. array with n_files, n_features dimensions
    """
    assert(len(n_blocks) == 3)

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
        x.append(prep_function(data))

    return np.array(x)


