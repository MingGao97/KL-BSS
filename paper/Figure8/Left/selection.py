import sys
sys.path.append('../..')
import os
os.makedirs('result', exist_ok=True)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from KLBSS import KLBSS

batches = [range(i*25,(i+1)*25) for i in range(8)] # 0-7
ds = [50,60,70,80,90] # 0-4

## data can be obtained from https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq
data = pd.read_csv('TCGA-PANCAN-HiSeq-801x20531/data.csv').iloc[:,1:]
variances = data.var(axis=0).values
dat = data.values[:,variances!=0]
variances = dat.var(axis=0)

ridx = np.where(variances > 21)[0].tolist()
ridx = [x for _, x in sorted(zip(variances[variances>21], ridx))]
rest = np.where(variances <= 21)[0].tolist()
variances_rest = variances[variances<=21]

Sstar = [i*2 for i in range(1,11)]
s = len(Sstar)
betamin = 0.2
ns = np.arange(200,801,100)

for d in ds:
    nums, bins, _ = plt.hist(variances_rest, bins=d-len(ridx))
    bins[0] = 0.01
    bins[-1] += 1
    for b, batch in enumerate(batches):
        for i in batch:
            # sample variable indicators
            xidx = []
            for k in range(d - len(ridx)):
                candid = np.where((variances_rest > bins[k]) & (variances_rest <= bins[k+1]))[0]
                xidx.append(rest[np.random.choice(candid)])
            xidx += ridx
            # centralize
            Xo = dat[:,xidx]
            Xo =Xo - Xo.mean(axis=0)

            # generate Y
            nn, _ = Xo.shape
            betas = betamin * np.ones(d)
            beta = betamin * (2 * np.random.binomial(1, 0.5, s) - 1)
            rand_sort = np.arange(nn)
            np.random.shuffle(rand_sort)
            X = Xo[rand_sort,:]
            Y = X[:,Sstar] @ beta + np.random.randn(nn)
            # estimate
            for j, n in enumerate(ns):
                print(i,j)
                Xt, Yt = X[:n,:], Y[:n]

                res_BSS = KLBSS(Xt,Yt,s=s,bss=True,mip=True,hard_gap=1e-5)
                res_klBSS = KLBSS(Xt,Yt,s=s,betam=betamin,mip=True,hard_gap=1e-5)

                f_BSS = open(f'result/Shat_BSS_{d}_{b}.txt', 'a')
                f_BSS.write(str(res_BSS) + '\n')
                f_BSS.close()

                f_klBSS = open(f'result/Shat_klBSS_{d}_{b}.txt', 'a')
                f_klBSS.write(str(res_klBSS) + '\n')
                f_klBSS.close()
