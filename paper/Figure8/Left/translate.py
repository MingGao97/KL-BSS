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


ns = np.arange(200,801,100)
N = 200
nb = 8

ds = [50,60,70,80,90]
methods = ['BSS','klBSS']


S = [i*2 for i in range(1,11)]
for d in ds:
    for metho in methods:
        res = np.zeros((4, N, len(ns)))
        for b in range(nb):
            with open(f'result/Shat_{metho}_{d}_{b}.txt') as f:
                Shatb = [literal_eval(line.rstrip()) for line in f]
            for i in range(25):
                for j in range(len(ns)):
                    Shat = Shatb[i*len(ns)+j]
                    res[0,b*25+i,j], res[1,b*25+i,j], res[2,b*25+i,j], res[3,b*25+i,j] = metric(S,Shat)
        np.savetxt(f'result/rec_{metho}_{d}.csv', res[0,:,:])
        np.savetxt(f'result/hd_{metho}_{d}.csv', res[1,:,:])
        np.savetxt(f'result/fdr_{metho}_{d}.csv', res[2,:,:])
        np.savetxt(f'result/tpr_{metho}_{d}.csv', res[3,:,:])
