import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import *
from sklearn.multioutput import MultiOutputClassifier
from sklearn import metrics, model_selection
import h5py
import time

print("startup")

file = h5py.File("../data/preprocessed/reduced.hdf5", "r")

#read the data
print("loading training set...")
train = np.array(file.get("train_data"))
y = np.loadtxt("targets.csv",dtype=int, delimiter=',') # targets for train set


# Scaling
scaler = preprocessing.StandardScaler().fit(train)
train = scaler.transform(train)

#get the scores
print("computing the score")

for n in [500, 1000, 2000]:
    forest = RandomForestClassifier(criterion='entropy', n_jobs=1, n_estimators=n)
    regr = MultiOutputClassifier(forest, n_jobs=1)
    scorer = metrics.make_scorer(metrics.hamming_loss, greater_is_better=False, needs_proba=False)
    scores = model_selection.cross_val_score(regr, train, y, scoring=scorer, cv=5, n_jobs=-1)
    avg = np.average(scores)

    print("n tree: ", n,  "  score: ", avg)
