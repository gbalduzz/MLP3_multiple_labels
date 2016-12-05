import numpy as np

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


def block_sum(x, nb, use_block_length = False):
    """
    :param x: array to reduce
    :param nb: number of blocks per dimension. must divide exactly x.shape
    :return:  return block sums of the x array
    """
    assert(((np.array(x.shape) % nb == 0).all()))
    if use_block_length == False : l = np.array(x.shape) / nb
    else :
        l = nb
        nb = np.array(x.shape) / nb
    res = np.zeros(nb)
    for i0 in range(nb[0]) :
        for i1 in range(nb[1]):
            for i2 in range(nb[2]):
                res[i0,i1,i2] = np.sum(np.ndarray.flatten(  x[ i0*l[0] : (i0+1)*l[0], i1*l[1] : (i1+1)*l[1], i2*l[2] : (i2+1)*l[2]]))

    return res

def concatenate_hystogram(x, nbins=40): #or auto maybe?
    #res  = np.array(x.shape[0], nbins)
    res = []
    for i in range(x.shape[0]):
        hist, _ = np.histogram(x[i], bins=nbins)
        res = np.append(res, hist)
    return res

def crop(x, n_blocks):
    """"
    cuts the 3D x array so that it is divisible by n_blocks, should roughly center the brain.
    """
    # enlarge this untill the division is exact
    initial_limit = np.array([[19,154], [24,186], [8,152]])

    remainder = (n_blocks - ((initial_limit[:, 1] - initial_limit[:, 0]) % n_blocks)) % n_blocks
    limit = initial_limit
    limit[:, 0] = initial_limit[:, 0] - remainder / 2
    limit[:, 1] = initial_limit[:, 1] + remainder / 2 + (remainder % 2)

    return x[limit[0,0]:limit[0,1], limit[1,0]:limit[1,1], limit[2,0]:limit[2,1]]

def blocks(x,nb):
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

def block_sum_scheme(x, n_blocks):
    return np.ndarray.flatten(block_sum(crop(x, n_blocks), n_blocks, use_block_length=True))
