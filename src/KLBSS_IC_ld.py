import numpy as np
from scipy import optimize
from sklearn.linear_model import LinearRegression
import itertools
from random import shuffle

def all_combinations(any_list, ubs):
    return itertools.chain.from_iterable(
        itertools.combinations(any_list, i + 1)
        for i in range(ubs))


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

def BSS_IC(X,Y,ubs,cc):
    n, d = X.shape
    candidate = all_combinations(np.arange(d), ubs)
    best_so_far = np.inf
    for candid in candidate:
        candidl = list(candid)
        score = Delta1(X[:,candidl], Y) + len(candidl) * cc
        if score < best_so_far:
            best_so_far = score
            Shat = candidl
    return Shat

def Delta2(XSt, Yt, beta, w):
    n, _ = XSt.shape
    reg = LinearRegression(fit_intercept = False).fit(XSt, Yt)
    Sigmahat = XSt.T @ XSt / (n-w)
    return QP_all(Sigmahat, reg.coef_, beta)

def KLBSS_vanilla_IC(X,Y,betas,ubs,cc):
    n, d = X.shape
    candidate = all_combinations(np.arange(d), ubs)
    best_so_far = np.inf
    for candid in candidate:
        candidl = list(candid)
        score = Delta2(X[:,candidl], Y, betas[candidl], 0) \
            + Delta1(X[:,candidl], Y) + len(candidl) * cc
        if score < best_so_far:
            best_so_far = score
            Shat = candidl
    return Shat

def compare(S,T,X,Y,betas,cc):
    WW = np.intersect1d(S,T).tolist()
    w = len(WW)
    if w == 0:
        scoreS = Delta2(X[:,S], Y, betas[S], w) + Delta1(X[:,S], Y) + len(S) * cc
        scoreT = Delta2(X[:,T], Y, betas[T], w) + Delta1(X[:,T], Y) + len(T) * cc
    else:
        SS = np.setdiff1d(S,WW).tolist()
        TT = np.setdiff1d(T,WW).tolist()
        XS = X[:,SS]
        XT = X[:,TT]
        XW = X[:,WW]
        Yt = residual(XW, Y)
        if len(SS) != 0:
            XSt = np.apply_along_axis(lambda y: residual(XW, y), 0, XS)
            del2S = Delta2(XSt, Yt, betas[SS], w)
        else:
            del2S = 0
        if len(TT) != 0:
            XTt = np.apply_along_axis(lambda y: residual(XW, y), 0, XT)
            del2T = Delta2(XTt, Yt, betas[TT], w)
        else:
            del2T = 0
        scoreS = del2S + Delta1(X[:,S], Y) + len(S) * cc
        scoreT = del2T + Delta1(X[:,T], Y) + len(T) * cc
    # print(scoreS,scoreT)
    return np.argmin([scoreT, scoreS])

def KLBSS_simple_IC(X,Y,betas,ubs,cc):
    n, d = X.shape
    candidate = list(all_combinations(np.arange(d), ubs))
    shuffle(candidate)
    for i, candid in enumerate(candidate):
        if i == 0:
            Shat = candid[:]
        else:
            if compare(list(Shat), list(candid), X, Y, betas, cc) == 0:
                Shat = candid[:]
    return list(Shat)

def KLBSS_full_IC(X,Y,betas,ubs,cc,smat=False):
    n, d = X.shape
    candidate = list(all_combinations(np.arange(d), ubs))
    M = len(candidate)
    score_mat = np.zeros((M, M))
    for i, candid in enumerate(candidate):
        for j, rival in enumerate(candidate):
            if i < j:
                score_mat[i,j] = compare(list(candid), list(rival), X, Y, betas, cc)
                score_mat[j,i] = 1 - score_mat[i,j]
    if smat:
        return score_mat
    else:
        candid_idx = np.argmax(score_mat.sum(axis=1))
        for i, candid in enumerate(candidate):
            if i == candid_idx:
                return list(candid)