import numpy as np
from sklearn.model_selection import KFold

def CV_KLBSS(X,Y,s,betamins,K,KLBSS,M,betas0=None,
            time_limit=600, mip_gap=1e-5, 
            hard_limit=900, hard_gap=1e-9, 
            nthread=8):
    n,d = X.shape
    if betas0 is None:
        betas0 = np.ones(d)
    res = np.zeros(len(betamins))
    kf = KFold(n_splits=K)
    k = 0
    for train, test in kf.split(X,Y):
        Xtr, Ytr = X[train,:], Y[train]
        Xte, Yte = X[test,:], Y[test]
        k += 1
        for i, betat in enumerate(betamins):
            print(f"Fold {k} of {K}, beta {i+1} of {len(betamins)}")
            betamu = betat * betas0
            betaml = betat * betas0
            Shat = KLBSS(Ytr,Xtr,s,False,betamu,betaml,M,
                            time_limit,mip_gap,hard_limit,hard_gap,nthread)[0]
            betahat = np.linalg.inv(Xtr[:,Shat].T @ Xtr[:,Shat]) @ Xtr[:,Shat].T @ Ytr
            loss = np.square(Yte - Xte[:,Shat] @ betahat).mean()
            res[i] += loss
    betaminhat = betamins[np.argmin(res)]
    betamu = betaminhat * betas0
    betaml = betaminhat * betas0
    return KLBSS(Y,X,s,False,betamu,betaml,M,
                time_limit,mip_gap,hard_limit,hard_gap,nthread)[0]