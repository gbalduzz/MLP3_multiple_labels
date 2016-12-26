import numpy as np
from modules import file_IO, models, postprocess
#import time


print("loading training set...")
tf = file_IO.load_directory("../data/set_train/")
print("loaded data with shape ", tf.shape)
# center the data
tf -= 0.5

print("loading labels...")
y = np.loadtxt("targets.csv", dtype=int, delimiter=',')

# choose or main or baseline _model
batch = 50
model = models.main_model(tf.shape)
# Fit the model
model.fit(tf, y[:tf.shape[0],:], nb_epoch=10, batch_size=batch)

print('make predictions:')
del tf
test = file_IO.load_directory("../data../set_test/")
predicted = model.predict(test, batch_size=batch)
#save raw and formatted predictions
del test
np.savetxt(predicted, "predivcted_raw.csv")

postprocess.format(predicted, "prediction.csv")
print("saved predictions to prediction.csv")
