import sys
sys.path.append('../..')

import numpy as np
import pandas as pd
from KLBSS import KLBSS

## data can be obtained from https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq
data = pd.read_csv('TCGA-PANCAN-HiSeq-801x20531/data.csv').iloc[:,1:]
variances = data.var(axis=0).values
dat = data.values[:,variances!=0]
variances = dat.var(axis=0)
argvar = np.argmax(variances)
coefs = np.apply_along_axis(lambda x: np.corrcoef(x,dat[:,argvar])[0,1], 0, dat)
sorts = np.argsort(np.abs(coefs))

def OLSbeta(X,Y,S):
    XS = X[:,S]
    return np.linalg.inv(XS.T @ XS) @ XS.T @ Y


d = 50
s = 10
betamins = 10**(np.arange(-1,0.4,0.2))
N = 100
res = np.zeros((N,2))

for i in range(N):
    print(f'=================={i}====================')
    idx = np.random.choice(sorts[:-1], d)

    X, Y = dat[:,idx], dat[:,argvar]
    X = X - X.mean(axis=0)
    Y = Y - Y.mean()

    rand_sort = np.arange(801)
    np.random.shuffle(rand_sort)
    X = X[rand_sort,:]
    Y = Y[rand_sort]

    n0 = 600
    Xtr, Ytr, Xte, Yte = X[:n0], Y[:n0], X[n0:], Y[n0:]
    
    Sk = KLBSS(Xtr,Ytr,s=s,cv=True,betams=betamins,K=5,mip=True)
    Sb = KLBSS(Xtr,Ytr,s=s,bss=True,mip=True)
    betak = OLSbeta(Xtr,Ytr,Sk)
    betab = OLSbeta(Xtr,Ytr,Sb)
    res[i,0] = np.square(Yte - Xte[:,Sk] @ betak).mean()
    res[i,1] = np.square(Yte - Xte[:,Sb] @ betab).mean()



### plot
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1, figsize=(6,5))
ax.axline((0, 0), slope=1, linestyle='--', color='gray', label='y=x line', alpha=0.5, linewidth=3)
ax.set_ylim((3,20))
ax.set_xlim((3,20))
ax.scatter(res[:,0],res[:,1],alpha=0.5)
ax.legend()
ax.set_xlabel('Prediction error of KLBSS')
ax.set_ylabel('Prediction error of BSS')
ax.set_title('Prediction performance of ' + r'$\widehat{S}$' + ' on real data')
fig.savefig('Figure8right.pdf',bbox_inches='tight')