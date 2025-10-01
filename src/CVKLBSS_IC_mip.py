import numpy as np
from sklearn.model_selection import KFold

def CV_KLBSS_IC(X,Y,betamins,K,KLBSS_IC,ubs,
                IC='BIC',cc=None,M=10,betas0=None,
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
        k += 1
        if IC == 'BIC':
            cc_ins = np.log(len(train)) / len(train)
        elif IC == 'EBIC':
            cc_ins = np.log(d) / len(train)
        elif IC == 'Delta':
            cc_ins = cc

        Xtr, Ytr = X[train,:], Y[train]
        Xte, Yte = X[test,:], Y[test]
        for i, betat in enumerate(betamins):
            print(f"Fold {k} of {K}, beta {i+1} of {len(betamins)}")
            betamu = betat * betas0
            betaml = betat * betas0
            Shat = KLBSS_IC(Ytr,Xtr,ubs,cc_ins,False,betamu,betaml,M,
                            time_limit,mip_gap,hard_limit,hard_gap,nthread)[0]
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
    betamu = betaminhat * betas0
    betaml = betaminhat * betas0

    return KLBSS_IC(Y,X,ubs,cc_out,False,betamu,betaml,M,
                    time_limit,mip_gap,hard_limit,hard_gap,nthread)[0]