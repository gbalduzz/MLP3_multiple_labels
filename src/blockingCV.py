from modules import  preprocessing
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from modules import postprocess
from modules.load_data import load_data
from sklearn import metrics, model_selection
from  modules.kernels import min_histo

y = np.loadtxt("targets.csv") # targets for train set

## Select best number of blocks and bins with cross validation on a simple Random Forest.

def ComputeScore(n_blocks, n_bins):
    data = load_data(n_blocks, n_bins, "train")

    # Scaling
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)

    regr = RandomForestClassifier(n_jobs=1, oob_score=True, n_estimators=1000)
    scorer = metrics.make_scorer(metrics.log_loss, greater_is_better=False, needs_proba=True)

    scores = model_selection.cross_val_score(regr, data, y, scoring=scorer, cv=5, n_jobs=-1)

    return np.average(scores), np.var(scores)

nbins = [50,80,120]
nblocks = [7,9,11]
out= np.array([])

for bin in nbins:
    for block in nblocks:

        avg, var = ComputeScore(block, bin)
        print("blocks: ", block, "bins ", bin, "score ",avg)
        out = np.append(out, [block, bin, avg, var])

np.savetxt("scores3.txt", out, header="blocks, bins, score, var")
