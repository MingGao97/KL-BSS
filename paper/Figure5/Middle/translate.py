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
nb = 8

ubss = [3,4,5,6,7]
methods = ['BSS','klBSS_vanilla','klBSS_simple','klBSS_vanilla_cv','klBSS_simple_cv']


for ubs in ubss:
    for metho in methods:
        res = np.zeros((4, N, len(ns)))
        for b in range(nb):
            with open(f'data_S/S_eff-ubs_{ubs}_{b}.txt') as f:
                SSb = [literal_eval(line.rstrip()) for line in f]
            with open(f'result/Shat_{metho}_eff-ubs_{ubs}_{b}.txt') as f:
                Shatb = [literal_eval(line.rstrip()) for line in f]
            for i in range(25):
                S = SSb[i]
                for j in range(len(ns)):
                    Shat = Shatb[i*8+j]
                    res[0,b*25+i,j], res[1,b*25+i,j], res[2,b*25+i,j], res[3,b*25+i,j] = metric(S,Shat)
        np.savetxt(f'result/rec_{metho}_{ubs}.csv', res[0,:,:])
        np.savetxt(f'result/hd_{metho}_{ubs}.csv', res[1,:,:])
        np.savetxt(f'result/fdr_{metho}_{ubs}.csv', res[2,:,:])
        np.savetxt(f'result/tpr_{metho}_{ubs}.csv', res[3,:,:])



