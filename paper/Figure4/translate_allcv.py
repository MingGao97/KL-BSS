import numpy as np
from ast import literal_eval

def metric(S,Shat):
    S = sorted(S)
    Shat = sorted(Shat)
    rec = S == Shat
    TP = len(np.intersect1d(S,Shat))
    FP = len(Shat) - TP
    FN = len(S) - TP
    fdr = FP / (FP + TP) if len(Shat) !=0 else 0
    tpr = TP / (TP + FN) if len(S) !=0 else 0
    hd = FP + FN
    return rec, hd, fdr, tpr


ns = np.arange(1000,8001,1000)
N = 200
nb = 100
neach = int(N/nb)

graph_types = ['ER','SF','Complete']
methods = ['klBSS_vanilla', 'klBSS_simple']
ic_or_not = ['','ic']

for ic in ic_or_not:
    for gt in graph_types:
        for metho in methods:
            res = np.zeros((4, N, len(ns)))
            for b in range(nb):
                with open(f'data_S_cv_ld/S_{gt}_{b}.txt') as f:
                    SSb = [literal_eval(line.rstrip()) for line in f]
                with open(f'result_{ic}cv_ld/Shat_{metho}_{gt}_{b}.txt') as f:
                    Shatb = [literal_eval(line.rstrip()) for line in f]
                for i in range(neach):
                    S = SSb[i]
                    for j in range(len(ns)):
                        Shat = Shatb[i*len(ns)+j]
                        res[0,b*neach+i,j], res[1,b*neach+i,j], res[2,b*neach+i,j], res[3,b*neach+i,j] = metric(S,Shat)
            np.savetxt(f'result_{ic}cv_ld/rec_{metho}_{gt}.csv', res[0,:,:])
            np.savetxt(f'result_{ic}cv_ld/hd_{metho}_{gt}.csv', res[1,:,:])
            np.savetxt(f'result_{ic}cv_ld/fdr_{metho}_{gt}.csv', res[2,:,:])
            np.savetxt(f'result_{ic}cv_ld/tpr_{metho}_{gt}.csv', res[3,:,:])

methods = ['klBSS']

for ic in ic_or_not:
    for gt in graph_types:
        for metho in methods:
            res = np.zeros((4, N, len(ns)))
            for i in range(N):
                with open(f'data_S_cv_hd/S_{gt}_{i}.txt') as f:
                    SSb = [literal_eval(line.rstrip()) for line in f]
                S = SSb[0]
                for j in range(len(ns)):
                    with open(f'result_{ic}cv_hd/Shat_{metho}_{gt}_{i}_{j}.txt') as f:
                        Shatb = [literal_eval(line.rstrip()) for line in f]
                    Shat = Shatb[0]
                    res[0,i,j], res[1,i,j], res[2,i,j], res[3,i,j] = metric(S,Shat)

            np.savetxt(f'result_{ic}cv_hd/rec_{metho}_{gt}.csv', res[0,:,:])
            np.savetxt(f'result_{ic}cv_hd/hd_{metho}_{gt}.csv', res[1,:,:])
            np.savetxt(f'result_{ic}cv_hd/fdr_{metho}_{gt}.csv', res[2,:,:])
            np.savetxt(f'result_{ic}cv_hd/tpr_{metho}_{gt}.csv', res[3,:,:])
