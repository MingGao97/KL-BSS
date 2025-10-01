import numpy as np
import igraph as ig
import networkx as nx

def compute_caus_order(G):
    d = G.shape[0]
    remain = list(range(d))
    caus_order = np.empty(d, dtype = int)
    for i in range(d-1):
        root = min(np.where(G.sum(axis=0) == 0)[0])
        caus_order[i] = remain[root]
        del remain[root]
        G = np.delete(G, root, axis = 0)
        G = np.delete(G, root, axis = 1)
    caus_order[d-1] = remain[0]
    return caus_order

def find_pa(G, node):
    return np.where(G[:,node] == 1)[0]


def simulate_dag(d, s0=2, graph_type='ER', s=3, permute=True):
    '''Simulate random DAG with some expected number of edges.

    Parameters
    ----------
        d : int
            num of nodes
        s0 : int
            expected num of edges in ER or SF
        graph_type : str
            'ER', 'SF', 'Tree', 'MC', 'Bipartite', 'complete'
        s : int
            upper bound of in-degree in Bipartite

    Returns
    ----------
    B : np.array
        binary adj matrix of DAG
    '''
    max_num_edge = int(d * (d - 1) / 2)
    if graph_type == 'ER':
        # Erdos-Renyi
        edge_from, edge_to = np.nonzero(np.triu(np.ones(d), k = 1))
        edges = np.random.choice(len(edge_from), min(s0, max_num_edge), replace = False)
        edge_from = edge_from[edges]
        edge_to = edge_to[edges]
        B = np.zeros((d, d))
        B[edge_from, edge_to] = 1
        if permute:
            rand_sort = np.arange(d)
            np.random.shuffle(rand_sort)
            B = B[rand_sort, :]
            B = B[:, rand_sort]
        
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(min(s0, max_num_edge) / d)), directed=True)
        B = np.array(G.get_adjacency().data)
        rand_sort = np.arange(d)
        np.random.shuffle(rand_sort)
        B = B[rand_sort, :]
        B = B[:, rand_sort]

    elif graph_type == 'Tree':
        # Tree graph
        B = np.tril(nx.to_numpy_matrix(nx.generators.trees.random_tree(d)))

    elif graph_type == 'MC':
        # Markov chain
        B = np.eye(d, k = 1)
        if permute:
            rand_sort = np.arange(d)
            np.random.shuffle(rand_sort)
            B = B[rand_sort, :]
            B = B[:, rand_sort]
    
    elif graph_type == 'Bipartite':
        V = np.arange(d)
        idx = np.random.choice(np.arange(1,d),1)[0]
        V1 = V[:idx]
        V2 = V[idx:]
        qb = min(s,len(V1))
        B = np.zeros((d,d))
        for j in V2:
            npa = np.random.choice(np.arange(1,qb+1),1)[0]
            jpa = np.random.choice(V1,npa)
            B[jpa,j] = 1

        rand_sort = np.arange(d)
        np.random.shuffle(rand_sort)
        B = B[rand_sort, :]
        B = B[:, rand_sort]
    
    elif graph_type == 'Complete':
        B = np.ones((d,d))
        B = np.tril(B,-1)
        rand_sort = np.arange(d)
        np.random.shuffle(rand_sort)
        B = B[:, rand_sort]
        B = B[rand_sort, :]

    return B


def simulate_error(err_dist, n, std):
    if err_dist == 'Gaussian':
        return np.random.randn(n) * std
    elif err_dist == 't':
        return np.random.standard_t(3,n) / np.sqrt(3) * std
    elif err_dist == 'Laplace':
        return np.random.laplace(0,1,n) / np.sqrt(2) * std
    elif err_dist == 'unif':
        return np.random.uniform(-1,1,n) * np.sqrt(3) * std
    elif err_dist == 'mixed':
        choice_list = ['Gaussian', 't', 'Laplace', 'unif', 'mixed']
        err_d = choice_list[np.random.choice(4,1)[0]]
        return simulate_error(err_d, n, std)
    else:
        print('No such distribution!')


def simulate_data(G, n, s, betamin, betamax_SEM, betamin_SEM,
                   sigma=1, sigmamin=0.5, sigmamax=1, 
                   err_dist='Gaussian'):
    '''simulate data given graph. 

    Parameters
    ----------
    G : np.array
        DAG adjacency matrix
    n : int
        sample size
    s : int
        sparsity level
    betamin : float
        linear coefficients in regression model
    betamax_SEM : float
        max linear coefficients in SEM
    betamin_SEM : float
        min linear coefficients in SEM
    sigma : float
        noise standard deviation
    err_dist : str
        noise distribution. 'Gaussian', 't', 'Laplace', 'unif', 'mixed'.

    Returns
    ----------
    X, Y : np.array
        data matrix
    '''
    # generate X
    d = G.shape[0]
    sigmas = np.random.uniform(sigmamin, sigmamax, d)
    X = np.empty((n,d))
    caus_order = compute_caus_order(G)
    for node in caus_order:
        pa_of_node = find_pa(G, node)
        epsilon_node = simulate_error(err_dist, n, sigmas[node])
        if len(pa_of_node) == 0:
            X[:,node] = epsilon_node
        else:
            beta = np.random.uniform(betamin_SEM, betamax_SEM, len(pa_of_node))
            beta *= (2 * np.random.binomial(1, 0.5, len(pa_of_node)) - 1)
            fpa = X[:,pa_of_node] @ beta
            X[:,node] = fpa + epsilon_node
    # generate Y
    S = np.sort(np.random.choice(d, s, replace=False))
    beta = betamin * (2 * np.random.binomial(1, 0.5, s) - 1)
    Y = X[:,S] @ beta + simulate_error(err_dist, n, sigma)
    return X, Y, S.tolist()



def simulate_data_indep_covar(d,n,q,betamin,sigma_z,sigma=1):
    X = np.random.randn(n*d).reshape(n,d)
    X[:,q:] *= sigma_z
    Y = X[:,:q].sum(axis=1) * betamin + np.random.randn(n) * sigma
    return X, Y
    