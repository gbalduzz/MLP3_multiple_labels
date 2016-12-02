import numpy as np

def min_histo(X,Y):
    def prod(x,y):
        return np.sum(np.amin(np.row_stack((x,y)), axis=0))

    print("computing kernel...")
    res = np.zeros([X.shape[0], Y.shape[0]])
    if X.shape != Y.shape or (X[0] != Y[0]).any() : # X != Y
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res[i,j] = prod(X[i], Y[j])
    else :  # symmetric case
        for i in range(res.shape[0]):
            for j in range(i,res.shape[1]):
                res[j,i] = res[i, j] = prod(X[i], Y[j])

    print("kernel computed.")
    return res
