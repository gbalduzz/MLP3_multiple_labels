import numpy as np

def format(data, filename='final_prediction.csv'):
    tag = np.array(range(1,len(data)+1), dtype=np.int32)
    mytype = np.dtype([('ID', np.int32), ('prediction', np.float64)])
    out = np.rec.fromarrays((tag,data), dtype=mytype)
    np.savetxt(filename, out, fmt=["%.d","%.3f"],
               delimiter=',', header="ID,prediction", comments='')
