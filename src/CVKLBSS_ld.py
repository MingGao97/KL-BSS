import numpy as np
from sklearn.model_selection import KFold

def CV_KLBSS(X,Y,s,betamins,K,KLBSS,betas0=None):
    n,d = X.shape
    if betas0 is None:
        betas0 = np.ones(d)
    res = np.zeros(len(betamins))
    kf = KFold(n_splits=K)
    for train, test in kf.split(X,Y):
        Xtr, Ytr = X[train,:], Y[train]
        Xte, Yte = X[test,:], Y[test]
        for i, betat in enumerate(betamins):
            betas = betat * betas0
            Shat = KLBSS(Xtr,Ytr,betas,s)
            betahat = np.linalg.inv(Xtr[:,Shat].T @ Xtr[:,Shat]) @ Xtr[:,Shat].T @ Ytr
            loss = np.square(Yte - Xte[:,Shat] @ betahat).mean()
            res[i] += loss
    betaminhat = betamins[np.argmin(res)]
    betas = betaminhat * betas0
    return KLBSS(X,Y,betas,s)