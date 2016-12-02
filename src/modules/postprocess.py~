import numpy as np

data = np.loadtxt("prediction.csv", dtype=np.float64)
id = np.array(range(1,138+1), dtype=np.int32)
mytype = np.dtype([('ID', np.int32), ('prediction', np.float64)])
out = np.rec.fromarrays((id,data), dtype=mytype)
print(out)
print(out[0][0].dtype)
np.savetxt("final_prediction.csv", out, fmt=["%.d","%.2f"],
           delimiter=',', header="ID,prediction", comments='')
