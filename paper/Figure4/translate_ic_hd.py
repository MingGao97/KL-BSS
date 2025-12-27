import numpy as np
from ast import literal_eval


def metric(S,Shat):
    S = sorted(S)
    Shat = sorted(Shat)
    rec = S == Shat
    TP = len(np.intersect1d(S,Shat))
    FP = len(Shat) - TP
    FN = len(S) - TP
    fdr = FP / (FP + TP) if len(Shat) !=0 else 0
    tpr = TP / (TP + FN) if len(S) !=0 else 0
    hd = FP + FN
    return rec, hd, fdr, tpr



ds = [50]
ss = [10, 15, 20] # 0-2
s0s = [1,2,4]  # 0-2
graph_types = ['ER','SF','Bipartite','Complete'] # 0-3
err_dists = ['Gaussian', 't', 'Laplace', 'unif', 'mixed'] # 0-4
batches = [range(i*25,(i+1)*25) for i in range(8)] # 0-7
methods = ['BSS','klBSS']
ns = np.arange(1000,8001,1000)
N = 200
nb = 8


ICs = ['BIC', 'EBIC', 'Delta']


def translate(string):
    for metho in methods:
        res = np.zeros((4, N, len(ns)))
        for b in range(nb):
            with open('data_S/S_' + string + f'_{b}.txt') as f:
                SSb = [literal_eval(line.rstrip()) for line in f]
            with open(f'result_IC/{IC}/Shat_{metho}_' + string + f'_{b}.txt') as f:
                Shatb = [literal_eval(line.rstrip()) for line in f]
            for i in range(25):
                S = SSb[i]
                for j in range(len(ns)):
                    Shat = Shatb[i*8+j]
                    res[0,b*25+i,j], res[1,b*25+i,j], res[2,b*25+i,j], res[3,b*25+i,j] = metric(S,Shat)
        np.savetxt(f'result_IC/{IC}/rec_' + metho + '_' + string + '.csv', res[0,:,:])
        np.savetxt(f'result_IC/{IC}/hd_' + metho + '_' + string + '.csv', res[1,:,:])
        np.savetxt(f'result_IC/{IC}/fdr_' + metho + '_' + string + '.csv', res[2,:,:])
        np.savetxt(f'result_IC/{IC}/tpr_' + metho + '_' + string + '.csv', res[3,:,:])



d = 50
for IC in ICs:
    for s in ss:
        for err_dist in err_dists:
            for graph_type in graph_types:
                if graph_type in ['ER','SF']:
                    for s0 in s0s:
                        string = f'{graph_type}-{s0}_{err_dist}_{d}_{s}'
                        translate(string)
                else:
                    string = f'{graph_type}_{err_dist}_{d}_{s}'
                    translate(string)


