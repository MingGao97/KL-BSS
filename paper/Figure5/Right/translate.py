import numpy as np

N = 50
for dim in ['low','high']:
    ds = np.arange(20,110,10).tolist() if dim=='low' else [200,500,1000]
    for mtrc in ['rec','hd','fdr','tpr','res_time']:
        out = np.zeros((N,len(ds)))
        for b in range(5):
            res = np.loadtxt(f'{mtrc}_{dim}_{b}.csv')
            out[10*b:10*(b+1)] = res[10*b:10*(b+1)]
        np.savetxt(f'res/{mtrc}_{dim}.csv', out)

        out = np.zeros((N,len(ds)))
        for b in range(5):
            res = np.loadtxt(f'{mtrc}_{dim}_{b}_BSS.csv')
            out[10*b:10*(b+1)] = res[10*b:10*(b+1)]
        np.savetxt(f'res/{mtrc}_{dim}_BSS.csv', out)

