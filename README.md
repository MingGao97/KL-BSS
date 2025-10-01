# KL-BSS: Neighbourhood selection/Support recovery/Variable selection in structural equation models (SEMs)

This is a `python` implementation of the following paper:

[1] Gao, M., Tai, W.M., Aragam, B. (2025). [KL-BSS: Rethinking optimality for neighbourhood selection in structural equation models](https://arxiv.org/abs/2306.02244)

## Requiments
- Python
- `scipy`
- `sklearn`
- `itertools`
- `gurobipy`
- `random`
- License of Gurobi is needed for mixed integer programming (MIP)

## Contents
- `KLBSS.py`: Main function to call KL-BSS
- `src`: Folder containing components of KL-BSS under different setups
- `utils.py`: Some helper functoins to simulate data and evaluate performances

## Demo
Generate a ER-1 DAG `G` with `d=10` nodes. Generate an SEM by `G` as `X` with coefficients uniformly sampled. Generate `Y` using `X` with sparsity being `s=4`. If sparsity is unknown, a upper bound is `ubs=5` is given. All noise variables are chosen from mixed distributions, i.e. randomly chosen from Gaussian, t, Laplace, uniform distributions. Sample `n=5000` data point.
```python
from KLBSS import KLBSS
from utils import *

n = 5000
d = 10
s = 4
ubs = 5
s0 = 1
graph_type = 'ER'
err_dist = 'mixed'

betamax_SEM = 5
betamin_SEM = 0.1
betamin = 0.1

G = simulate_dag(d, s0*d, graph_type, s)
X, Y, S = simulate_data(G, n, s, betamin, betamax_SEM, betamin_SEM, err_dist=err_dist)

print(S)
```

Best subset selection (BSS) with known sparsity
```python
KLBSS(X,Y,bss=True,s=s)
```

BSS with unknown sparsity (BIC)
```python
KLBSS(X,Y,bss=True,ubs=ubs,ic='BIC')
```

BSS with known sparsity and MIP
```python
KLBSS(X,Y,bss=True,s=s,mip=True)
```

BSS with unknown sparsity (BIC) and MIP
```python
KLBSS(X,Y,bss=True,ubs=ubs,ic='BIC',mip=True)
```

vanilla KL-BSS with known sparsity and betamin 
```python
KLBSS(X,Y,klbss_type='vanilla',s=s,betam=betamin)
```

KL-BSS with known sparsity and betamin
```python
KLBSS(X,Y,klbss_type='simple',s=s,betam=betamin)
```

vanilla KL-BSS with known sparsity and 5-fold CV for betamin
```python
betamins = 10**(np.arange(-2.4,0.8,0.2))
KLBSS(X,Y,klbss_type='vanilla',s=s,cv=True,betams=betamins,K=5)
```

KL-BSS with known sparsity and 5-fold CV for betamin
```python
KLBSS(X,Y,klbss_type='simple',s=s,cv=True,betams=betamins,K=5)
```

vanilla KL-BSS with unknown sparsity (BIC) and betamin
```python
KLBSS(X,Y,klbss_type='vanilla',ubs=ubs,ic='BIC',betam=betamin)
```

KL-BSS with unknown sparsity (BIC) and betamin
```python
KLBSS(X,Y,klbss_type='simple',ubs=ubs,ic='BIC',betam=betamin)
```

vanilla KL-BSS with unknown sparsity (BIC) and 5-fold CV for betamin
```python
KLBSS(X,Y,klbss_type='vanilla',ubs=ubs,ic='BIC',cv=True,betams=betamins,K=5)
```

KL-BSS with unknown sparsity (BIC) and 5-fold CV for betamin
```python
KLBSS(X,Y,klbss_type='simple',ubs=ubs,ic='BIC',cv=True,betams=betamins,K=5)
```

KL-BSS with known sparsity and betamin and MIP
```python
KLBSS(X,Y,s=s,betam=betamin,mip=True)
```

KL-BSS with unknown sparsity (BIC) and betamin and MIP
```python
KLBSS(X,Y,ubs=ubs,ic='BIC',betam=betamin,mip=True)
```

KL-BSS with known sparsity and 5-fold CV for betamin and MIP
```python
KLBSS(X,Y,s=s,cv=True,betams=betamins,K=5,mip=True)
```

KL-BSS with unknown sparsity (BIC) and 5-fold CV for betamin and MIP
```python
KLBSS(X,Y,ubs=ubs,ic='BIC',cv=True,betams=betamins,K=5,mip=True)
```
