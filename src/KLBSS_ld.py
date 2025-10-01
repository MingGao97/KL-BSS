import numpy as np
from scipy import optimize
from sklearn.linear_model import LinearRegression
from itertools import combinations
from scipy.special import comb

def QP(Sigma, gamma, beta, sign):
  x0 = beta*sign
  def loss(x):
      return (x-gamma).T @ Sigma @ (x-gamma)
  def jac(x):
      return 2 * Sigma @ (x-gamma)
  cons = {'type':'ineq',
          'fun':(lambda x: sign*x - beta),
          'jac':(lambda x: sign * np.eye(len(gamma)))}
  opt = {'disp':False}
  res_cons = optimize.minimize(loss,
                               x0,
                               jac=jac,
                               constraints=cons,
                               method='SLSQP',
                               options=opt)
  return res_cons.fun

def pm(n):
    nn = 1 << n
    rtn = []
    for i in range(nn):
        x = [1] * n
        for j in range(n):
            if i & (1<<j) != 0:
                x[j] = -1
        rtn.append(x)
    return rtn

def QP_all(Sigma, gamma, beta):
    res = []
    for sign in pm(len(gamma)):
        res.append(QP(Sigma, gamma, beta, sign))
    return min(res)


def norm(vec):
    return np.square(vec).sum()

def residual(XS, Y):
    n, q = XS.shape
    reg = LinearRegression(fit_intercept = False).fit(XS, Y)
    return Y - reg.predict(XS)

def Delta1(XS, Y):
    n, q = XS.shape
    resid = residual(XS, Y)
    return norm(resid) / (n-q)

def BSS(X,Y,q):
    n, d = X.shape
    candidate = combinations(np.arange(d), q)
    best_so_far = np.inf
    for candid in candidate:
        candidl = list(candid)
        score = Delta1(X[:,candidl], Y)
        if score < best_so_far:
            best_so_far = score
            Shat = candidl
    return Shat

def Delta2(XSt, Yt, beta, q):
    n, l = XSt.shape
    reg = LinearRegression(fit_intercept = False).fit(XSt, Yt)
    Sigmahat = XSt.T @ XSt / (n-(q-l))
    return QP_all(Sigmahat, reg.coef_, beta)

def KLBSS_vanilla(X,Y,betas,q):
    n, d = X.shape
    candidate = combinations(np.arange(d), q)
    best_so_far = np.inf
    for candid in candidate:
        candidl = list(candid)
        score = Delta2(X[:,candidl], Y, betas[candidl], q) \
            + Delta1(X[:,candidl], Y)
        if score < best_so_far:
            best_so_far = score
            Shat = candidl
    return Shat

def compare(S,T,X,Y,betas):
    q = len(S)
    WW = np.intersect1d(S,T).tolist()
    if len(WW) == 0:
        scoreS = Delta2(X[:,S], Y, betas[S], q) + Delta1(X[:,S], Y)
        scoreT = Delta2(X[:,T], Y, betas[T], q) + Delta1(X[:,T], Y)
    else:
        SS = np.setdiff1d(S,WW).tolist()
        TT = np.setdiff1d(T,WW).tolist()
        XS = X[:,SS]
        XT = X[:,TT]
        XW = X[:,WW]
        XSt = np.apply_along_axis(lambda y: residual(XW, y), 0, XS)
        XTt = np.apply_along_axis(lambda y: residual(XW, y), 0, XT)
        Yt = residual(XW, Y)
        scoreS = Delta2(XSt, Yt, betas[SS], q) + Delta1(X[:,S], Y)
        scoreT = Delta2(XTt, Yt, betas[TT], q) + Delta1(X[:,T], Y)
    # print(scoreS,scoreT)
    return np.argmin([scoreT, scoreS])

def KLBSS_simple(X,Y,betas,q,init_order=False):
    n, d = X.shape
    M = comb(d, q, exact=True)

    if init_order:
        # adversarial!
        perm = np.arange(d)
    else:
        perm = np.random.permutation(d)
    
    Xperm = X[:,perm]
    for i, candid in enumerate(combinations(np.arange(d), q)):
        # print(i)
        if i == 0:
            Shat = candid[:]
        else:
            if compare(list(Shat), list(candid), Xperm, Y, betas) == 0:
                Shat = candid[:]
    return sorted(list(set(perm[list(Shat)])))

def KLBSS_full(X,Y,betas,q,smat=False):
    n, d = X.shape
    M = comb(d, q, exact=True)
    score_mat = np.zeros((M, M))
    for i, candid in enumerate(combinations(np.arange(d), q)):
        for j, rival in enumerate(combinations(np.arange(d), q)):
            # print(i,j)
            if i < j:
                score_mat[i,j] = compare(list(candid), list(rival), X, Y, betas)
                score_mat[j,i] = 1 - score_mat[i,j]
    if smat:
        return score_mat
    else:
        candid_idx = np.argmax(score_mat.sum(axis=1))
        for i, candid in enumerate(combinations(np.arange(d), q)):
            if i == candid_idx:
                return list(candid)
