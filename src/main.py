import numpy as np
from modules import file_IO, models
#import time


print("loading training set...")
tf = file_IO.load_directory("../data/set_train/")
# center the data
tf -= 0.5

print("loading labels...")
y = np.loadtxt("targets.csv", dtype=int, delimiter=',')

## Reshape array according to tensorflow requirements
tf = tf.reshape(tf.shape[0],tf.shape[1],tf.shape[2],tf.shape[3],1)
num_classes = 3 #No of outputs

"""For the model below, the input shape needs to be specified. At present it is
(176,208,176,1), but if you chose to resize the image, it should be changed accordingly.
The first argument of the convolution layer is the number of filters(32 below), the next is
the size of kernel(3x3x3 in this case)"""



# choose or main or baseline _model
model = models.main_model()
# Fit the model
model.fit(tf, y, nb_epoch=10, batch_size=3)


#postprocess.format(prediction, "prediction.csv")
#print("saved predictions to prediction.csv")
