import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from modules import postprocess
import h5py
import time

outname ="prediction.csv"

print("startup")

file = h5py.File("../data/preprocessed/reduced.hdf5", "r")

#read the data
print("loading training set...")
train = np.array(file.get("train_data"))
y = np.loadtxt("targets.csv", dtype=int, delimiter=',') # targets for train set



# Train the Model
start = time.clock()
print("start training")
forest = RandomForestClassifier(n_estimators=50000, criterion='entropy')
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
multi_target_forest.fit(train,y)
finish = time.clock()
print("training time: ", finish-start)
del train


# Make Predictions and save
print("loading and making predictions")
test  = np.array(file.get("test_data"))
file.close()

prediction = multi_target_forest.predict(test)
postprocess.format(prediction, outname)
print("saved predictions to ",outname)
