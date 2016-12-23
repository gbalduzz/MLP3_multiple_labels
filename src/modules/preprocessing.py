import numpy as np
import h5py
import multiprocessing

def crop(x, n_blocks, bounds):
    """"
    cuts the 3D x array so that it is divisible by n_blocks, should roughly center the brain.
    """
    # enlarge this untill the division is exact

    remainder = (n_blocks - ((bounds[:, 1] - bounds[:, 0]) % n_blocks)) % n_blocks
    limit = bounds
    limit[:, 0] = bounds[:, 0] - remainder / 2
    limit[:, 1] = bounds[:, 1] + remainder / 2 + (remainder % 2)

    return x[limit[0,0]:limit[0,1], limit[1,0]:limit[1,1], limit[2,0]:limit[2,1]]

def blocks(x,nb, bounds):
    x = crop(x, nb, bounds)
    assert (((np.array(x.shape) % nb == 0).all()))
    l = np.array(x.shape) / nb
    res = np.zeros([np.prod(nb),np.prod(l)])
    count = 0
    for i0 in range(nb[0]):
        for i1 in range(nb[1]):
            for i2 in range(nb[2]):
                res[count] = np.ndarray.flatten(
                    x[i0 * l[0]: (i0 + 1) * l[0], i1 * l[1]: (i1 + 1) * l[1],
                    i2 * l[2]: (i2 + 1) * l[2]])
                count += 1
    return res

def remove_zero_columns(x,n_train=-1,tollerance=0):
    """
    :param x: array to reduce, assumed positive
    :param tollerance: maximum abs value that is considered zero
    :return: np.array with removed columns
    """
    #max_values = np.amax(x[:n_train,:], axis=0)
    max_values = np.amax(x, axis=0)
    mask = np.ones(max_values.shape)*tollerance
    keep_indices = np.greater(max_values,mask)
    return x[:, keep_indices]



def compute_pca(x, n, scale = False):
    """
    :param x: input dataset
    :param n: number if dimensions to keep
    :return:  reduced dataset, change if basis matrix, eingenvalues
    """
    epsilon = 1e-4
    n_sets = x.shape[0]
    x -= np.mean(x,  axis = 0)
    cov = np.dot(x.T, x) / n_sets
    U,S,V = np.linalg.svd(cov)
    #reduce dimensions
    if scale: return np.dot(x, U[:,:n]) / np.sqrt(S[:n]+epsilon)
    else : return np.dot(x, U[:,:n])


def blocked_pca(data, blocks, k, file):
    assert(data.shape[1] % blocks == 0)
    block_size = data.shape[1] / blocks

    #read if previous work was done
    try:
        last_done = np.array(file.get("last_done"))[0]
    except:
        file.create_dataset("last_done", shape = [1], data=0)
        last_done = -1

    for i in range(last_done+1, blocks):
        slice = data[:, i*block_size:(i+1)*block_size]
        res = compute_pca(slice, k, scale=False)
        # write result to file
        file.create_dataset("block_"+str(i), data=res)
        #update iteration count
        set = file["last_done"]
        set[...] = i
        file.flush()
        print("done step: ",i)


