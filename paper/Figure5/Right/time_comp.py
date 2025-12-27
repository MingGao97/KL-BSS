import sys
sys.path.append('../..')
import os
os.makedirs('res', exist_ok=True)

import numpy as np
from KLBSS import KLBSS
from utils import *
import time

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


s = 10
s0 = 2
graph_type = 'ER'
err_dist = 'Gaussian'
N = 50
n = 5000
betamax_SEM = 2
betamin_SEM = 0.1
betamin = 0.1

dims = ['low','high']
batches = [range(i*10,(i+1)*10) for i in range(5)] # 0-4
hard_gap = 1e-2

for dim in dims:
    ds = np.arange(20,110,10).tolist() if dim=='low' else [200,500,1000]
    for b, batch in enumerate(batches):
        string = f'{dim}_{b}'
        string_bss = f'{dim}_{b}_BSS'
        res = np.zeros((4,N,len(ds)))
        res_time = np.zeros((N,len(ds)))
        res_bss = np.zeros((4,N,len(ds)))
        res_time_bss = np.zeros((N,len(ds)))
        for i in batch:
            for j, d in enumerate(ds):
                G = simulate_dag(d, s0*d, graph_type, s)
                X, Y, S = simulate_data(G, n, s, betamin, betamax_SEM, betamin_SEM, err_dist=err_dist)
                print(string, i,d)

                start = time.time()
                Shat = KLBSS(X,Y,s=s,betam=betamin,mip=True,hard_gap=1e-2,hard_limit=3600)
                time_klbss = time.time() - start
                start = time.time()
                Shat_bss= KLBSS(X,Y,s=s,bss=True,mip=True,hard_gap=1e-2,hard_limit=3600)
                time_bss = time.time() - start
                
                res[0,i,j], res[1,i,j], res[2,i,j], res[3,i,j] = metric(S,Shat)
                res_bss[0,i,j], res_bss[1,i,j], res_bss[2,i,j], res_bss[3,i,j] = metric(S,Shat_bss)

                np.savetxt(f'res/rec_{string}.csv', res[0,:,:])
                np.savetxt(f'res/hd_{string}.csv', res[1,:,:])
                np.savetxt(f'res/fdr_{string}.csv', res[2,:,:])
                np.savetxt(f'res/tpr_{string}.csv', res[3,:,:])
                np.savetxt(f'res/rec_{string_bss}.csv', res_bss[0,:,:])
                np.savetxt(f'res/hd_{string_bss}.csv', res_bss[1,:,:])
                np.savetxt(f'res/fdr_{string_bss}.csv', res_bss[2,:,:])
                np.savetxt(f'res/tpr_{string_bss}.csv', res_bss[3,:,:])
                
                res_time[i,j] = time_klbss
                res_time_bss[i,j] = time_bss
                np.savetxt(f'res/res_time_{string}.csv', res_time)
                np.savetxt(f'res/res_time_{string_bss}.csv', res_time_bss)
            





