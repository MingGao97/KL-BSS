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
nb = 40
neach = int(N/nb)

ss = [10,15,20]
methods = ['klBSS','BSS']


for s in ss:
    for metho in methods:
        res = np.zeros((4, N, len(ns)))
        for b in range(nb):
            with open(f'data_S_randsupp/S_{s}_{b}.txt') as f:
                SSb = [[int(x) for x in line.rstrip().strip('[]').split()] for line in f]
            with open(f'result_hd_randsupp/Shat_{metho}_{s}_{b}.txt') as f:
                Shatb = [literal_eval(line.rstrip()) for line in f]
            for i in range(neach):
                S = SSb[i]
                for j in range(len(ns)):
                    Shat = Shatb[i*len(ns)+j]
                    res[0,b*neach+i,j], res[1,b*neach+i,j], res[2,b*neach+i,j], res[3,b*neach+i,j] = metric(S,Shat)
        np.savetxt(f'result_hd_randsupp/rec_{metho}_{s}.csv', res[0,:,:])
        np.savetxt(f'result_hd_randsupp/hd_{metho}_{s}.csv', res[1,:,:])
        np.savetxt(f'result_hd_randsupp/fdr_{metho}_{s}.csv', res[2,:,:])
        np.savetxt(f'result_hd_randsupp/tpr_{metho}_{s}.csv', res[3,:,:])


for s in ss:
    res = np.zeros((4, N, len(ns)))
    for i in range(N):
        with open(f'data_S_randsupp/S_{s}_{i//5}.txt') as f:
            SSb = [[int(x) for x in line.rstrip().strip('[]').split()] for line in f]
        S = SSb[i%5]
        for j in range(len(ns)):
            with open(f'result_cv_hd_randsupp/Shat_klBSS_{s}_{i}_{j}.txt') as f:
                Shatb = [literal_eval(line.rstrip()) for line in f]
            Shat = Shatb[0]
            res[0,i,j], res[1,i,j], res[2,i,j], res[3,i,j] = metric(S,Shat)

    np.savetxt(f'result_cv_hd_randsupp/rec_klbss_{s}.csv', res[0,:,:])
    np.savetxt(f'result_cv_hd_randsupp/hd_klbss_{s}.csv', res[1,:,:])
    np.savetxt(f'result_cv_hd_randsupp/fdr_klbss_{s}.csv', res[2,:,:])
    np.savetxt(f'result_cv_hd_randsupp/tpr_klbss_{s}.csv', res[3,:,:])