import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt


print("startup")

filename = "../data/set_train/train_12.nii"

img = nb.load(filename).get_data()[:,:,:,0]

plt.imshow(img[img.shape[0]/2,:,:], cmap='gray')
plt.show()
