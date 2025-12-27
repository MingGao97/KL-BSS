import sys
sys.path.append('../..')
import os
os.makedirs('data_S', exist_ok=True)
os.makedirs('result', exist_ok=True)

import numpy as np
from utils import *
from KLBSS import KLBSS

batches = [range(i*25,(i+1)*25) for i in range(8)] # 0-7
d = 7
s = 3
s0 = 2
graph_type='SF'
err_dist = 'Gaussian'
betamin = 0.1
betamax = 5
ns = np.arange(1000,8001,1000)

# candidate betamins
betamins = 10**(np.arange(-2.4,0.8,0.2))
K=5

for b, batch in enumerate(batches):
    string = f'CV_{b}'
    for i in batch:
        G = simulate_dag(d, s0*d, graph_type, s)
        X, Y, S = simulate_data(G, max(ns), s, betamin, betamax, betamin, err_dist=err_dist)
        f = open('data_S/S_' + string + '.txt', 'a')
        f.write(str(S) + '\n')
        f.close()
        for j, n in enumerate(ns):
            print(f'N: {i}; n: {n}')
            Xt, Yt = X[:n,:], Y[:n]

            res_BSS = KLBSS(Xt,Yt,s=s,bss=True)
            res_klBSS_vanilla = KLBSS(Xt,Yt,s=s,klbss_type='vanilla',betam=betamin)
            res_klBSS_simple = KLBSS(Xt,Yt,s=s,klbss_type='simple',betam=betamin)

            res_klBSS_vanilla_cv = KLBSS(Xt,Yt,s=s,cv=True,betams=betamins,K=K,klbss_type='vanilla')
            res_klBSS_simple_cv = KLBSS(Xt,Yt,s=s,cv=True,betams=betamins,K=K,klbss_type='simple')

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

            for ll, betamt in enumerate(betamins):
                betas_t = betamt * np.ones(d)
                res_klBSS_simple_t = KLBSS(Xt,Yt,s=s,betam=betas_t,klbss_type='simple')
                f = open(f'result/Shat_klBSS_simple_{ll}_' + string + '.txt', 'a')
                f.write(str(res_klBSS_simple_t) + '\n')
                f.close()
        

