from modules import  preprocessing
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from modules import postprocess
import h5py
import time
from  modules.kernels import min_histo

print("startup")

file = h5py.File("preprocessed/reduced.hdf5", "r")

#read the data
print("loading training set...")
train = np.array(file.get("train_data"))

y = np.loadtxt("targets.csv") # targets for train set


# Scaling
scaler = preprocessing.StandardScaler().fit(train)
train = scaler.transform(train)

# Train the Model
start = time.clock()
print("start training")
regr = RandomForestClassifier(n_jobs=-1, n_estimators=10000, criterion='entropy')
regr.fit(train,y)
finish = time.clock()
print("training time: ", finish-start)
del train


# Make Predictions and save
print("loading and making predictions")
test  = np.array(file.get("test_data"))
file.close()
#test = append_ratio(test, "preprocessed/ratio_testing.csv")
test = scaler.transform(test)

prediction = regr.predict_proba(test)[:,1]
postprocess.format(prediction, "final_sub.csv")
print("saved predictions to final_sub.csv")
