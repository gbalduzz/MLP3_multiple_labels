import numpy as np
"""
def format(data, filename='final_prediction.csv'):
    tag = np.array(range(1,len(data)+1), dtype=np.int32)
    mytype = np.dtype([('ID', np.int32), ('prediction', np.float64)])
    out = np.rec.fromarrays((tag,data), dtype=mytype)
    np.savetxt(filename, out, fmt=["%.d","%.3f"],
               delimiter=',', header="ID,prediction", comments='')
"""
def _boolString(b):
    if(b): return 'True'
    else : return 'False'

def format(data, filename='final_prediction.csv'):
    mytype = np.dtype([('ID', np.int32), ('Sample', np.int32),
                       ('Label', np.str), ('Predicted', np.bool)])

    id = 0
    out =[]
    for sample in range(data.shape[0]):
        for label in ["gender","age","health"]:
            value = data[sample, id%3]
            out.append([str(id), str(sample), label, _boolString(value)])
            id += 1


    #out = np.rec.fromarrays((out), dtype=mytype)
    np.savetxt(filename, out,
               delimiter=',', header="ID,Sample,Label,Predicted", comments='')
