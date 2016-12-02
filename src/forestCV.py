from modules import  preprocessing
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, model_selection
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

#get the scores
print("computing the score")
regr = RandomForestClassifier(criterion='gini', n_jobs=1, n_estimators=1000)
scorer = metrics.make_scorer(metrics.log_loss, greater_is_better=False, needs_proba=True)
scores = model_selection.cross_val_score(regr, train, y, scoring=scorer, cv=5, n_jobs=-1)
avg = np.average(scores)

print("score ", avg)
