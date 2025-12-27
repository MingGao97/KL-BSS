import sys
sys.path.append('../..')
import os
os.makedirs('data_S', exist_ok=True)
os.makedirs('result', exist_ok=True)

import numpy as np
from KLBSS import KLBSS
from utils import *

ubss = [3,4,5,6,7]
batches = [range(i*25,(i+1)*25) for i in range(8)] # 0-7
d = 7
s = 3
s0 = 2
graph_type='SF'
err_dist = 'Gaussian'
betamin = 0.1
betamax = 5
K = 5
betamins = 10**(np.arange(-2.4,0.8,0.2))
ns = np.arange(1000,8001,1000)

for ubs in ubss:
    for b, batch in enumerate(batches):
        string = f'eff-ubs_{ubs}_{b}'
        for i in batch:
            G = simulate_dag(d, s0*d, graph_type, s)
            X, Y, S = simulate_data(G, max(ns), s, betamin, betamax, betamin, err_dist=err_dist)
            f = open('data_S/S_' + string + '.txt', 'a')
            f.write(str(S) + '\n')
            f.close()
            for j, n in enumerate(ns):
                if graph_type in ['ER','SF']:
                    print(f'ubs: {ubs}; N: {i}; n: {n}')
                else:
                    print(f'ubs: {ubs}; N: {i}; n: {n}')
                Xt, Yt = X[:n,:], Y[:n]

                res_BSS = KLBSS(Xt,Yt,ubs=ubs,bss=True,ic='BIC')
                res_klBSS_vanilla = KLBSS(Xt,Yt,ubs=ubs,klbss_type='vanilla',betam=betamin,ic='BIC')
                res_klBSS_simple = KLBSS(Xt,Yt,ubs=ubs,klbss_type='simple',betam=betamin,ic='BIC')
                res_klBSS_vanilla_cv = KLBSS(Xt,Yt,ubs=ubs,klbss_type='vanilla',betams=betamins,cv=True,K=K,ic='BIC')
                res_klBSS_simple_cv = KLBSS(Xt,Yt,ubs=ubs,klbss_type='simple',betams=betamins,cv=True,K=K,ic='BIC')

                f_BSS = open(f'result/Shat_BSS_' + string + '.txt', 'a')
                f_BSS.write(str(res_BSS) + '\n')
                f_BSS.close()

                f_klBSS_vanilla = open(f'result/Shat_klBSS_vanilla_' + string + '.txt', 'a')
                f_klBSS_vanilla.write(str(res_klBSS_vanilla) + '\n')
                f_klBSS_vanilla.close()

                f_klBSS_simple = open(f'result/Shat_klBSS_simple_' + string + '.txt', 'a')
                f_klBSS_simple.write(str(res_klBSS_simple) + '\n')
                f_klBSS_simple.close()

                f_klBSS_vanilla_cv = open(f'result/Shat_klBSS_vanilla_cv_' + string + '.txt', 'a')
                f_klBSS_vanilla_cv.write(str(res_klBSS_vanilla_cv) + '\n')
                f_klBSS_vanilla_cv.close()

                f_klBSS_simple_cv = open(f'result/Shat_klBSS_simple_cv_' + string + '.txt', 'a')
                f_klBSS_simple_cv.write(str(res_klBSS_simple_cv) + '\n')
                f_klBSS_simple_cv.close()

