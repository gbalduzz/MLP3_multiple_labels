from modules import  preprocessing
import numpy as np
from sklearn import preprocessing, feature_selection
from sklearn.ensemble import RandomForestClassifier
from modules import postprocess
import matplotlib.pyplot as plt
import h5py
import time
from  modules.kernels import min_histo

print("startup")

file = h5py.File("preprocessed/reduced.hdf5", "r")

#read the data
print("loading training set...")
train = np.array(file.get("train_data"))

y = np.loadtxt("targets.csv", dtype=bool) # targets for train set


#select
selector = feature_selection.SelectKBest(k=2).fit(train, y)
train = selector.transform(train)
# Scaling
scaler = preprocessing.StandardScaler().fit(train)
#scaler = preprocessing.MaxAbsScaler().fit(train)
train = scaler.transform(train)

healty = train[y,:]
ill = train[-y,:]

plt.scatter(healty[:,0], healty[:,1],marker='o')
plt.scatter(ill[:,0], ill[:,1],marker='x')
plt.show()

"""
# Train the Model
start = time.clock()
print("start training")
regr = RandomForestClassifier(n_jobs=-1, oob_score=True, n_estimators=20000)
regr.fit(train,y)
finish = time.clock()
print("training time: ", finish-start)
del train

#print model parameters
#np.savetxt('wheights_output.csv', np.row_stack((regr.intercept_, regr.dual_coef_)))


# Make Predictions and save
print("loading and making predictions")
test  = np.array(file.get("test_data"))
file.close()
test = selector.transform(test)
test = scaler.transform(test)

prediction = regr.predict_proba(test)[:,1]
postprocess.format(prediction, "predictions.csv")
print("done")
"""
