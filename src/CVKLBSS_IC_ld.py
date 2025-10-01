import numpy as np
from sklearn.model_selection import KFold

def CV_KLBSS_IC(X,Y,betamins,K,KLBSS_IC,ubs,IC='BIC',cc=None,betas0=None):
    n,d = X.shape
    if betas0 is None:
        betas0 = np.ones(d)
    res = np.zeros(len(betamins))
    kf = KFold(n_splits=K)
    for train, test in kf.split(X,Y):

        if IC == 'BIC':
            cc_ins = np.log(len(train)) / len(train)
        elif IC == 'EBIC':
            cc_ins = np.log(d) / len(train)
        elif IC == 'Delta':
            cc_ins = cc

        Xtr, Ytr = X[train,:], Y[train]
        Xte, Yte = X[test,:], Y[test]
        for i, betat in enumerate(betamins):
            betas = betat * betas0
            Shat = KLBSS_IC(Xtr,Ytr,betas,ubs,cc_ins)
            betahat = np.linalg.inv(Xtr[:,Shat].T @ Xtr[:,Shat]) @ Xtr[:,Shat].T @ Ytr
            loss = np.square(Yte - Xte[:,Shat] @ betahat).mean()
            res[i] += loss
    
    if IC == 'BIC':
        cc_out = np.log(n) / n
    elif IC == 'EBIC':
        cc_out = np.log(d) / n
    elif IC == 'Delta':
        cc_out = cc

    betaminhat = betamins[np.argmin(res)]
    betas = betaminhat * betas0

    return KLBSS_IC(X,Y,betas,ubs,cc_out)