import numpy as np

def threeDindex(index):
    subind = np.zeros([3], dtype=int)
    #for i in range(3):
        #fast first
        #subind[i] = index % 7;
        #index = (index - subind[i]) / 7;

    #fast last
    subind[0] = index / 49
    index -= subind[0]*49
    subind[1] = index / 7
    subind[2] = index - subind[1]*7
    return subind

def keepN(idx):
    n1 = 420
    n2 = 42

    if (idx == 0).any() :
        return n2
    if (idx == 7-1).any():
        return n2
    else: return n1
