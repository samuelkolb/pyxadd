# https://github.com/networkx/networkx/blob/862cc2b4ed4526cd90f300ff3054bd922ff55969/networkx/algorithms/link_analysis/pagerank_alg.py

def pagerank_scipy(G, alpha=0.85, personalization=None,
                   max_iter=100, tol=1.0e-6, weight='weight',
                   dangling=None):
    import scipy.sparse
    import scipy.linalg

    G = G.T  # Transpose for consistency
    N = G.shape[0]
    if N == 0:
        return {}

    nodelist = list(range(N))
    M = scipy.sparse.coo_matrix(G, dtype=float)
    S = scipy.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    M = Q * M

    # initial vector
    x = scipy.repeat(1.0 / N, N)

    # Personalization vector
    if personalization is None:
        p = scipy.repeat(1.0 / N, N)
    else:
        p = scipy.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        p = p / p.sum()

    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = scipy.array([dangling.get(n, 0) for n in nodelist],
                                       dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = scipy.where(S == 0)[0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + \
            (1 - alpha) * p
        # check convergence, l1 norm
        err = scipy.absolute(x - xlast).sum()
        # err = scipy.linalg.norm(x - xlast)
        if err < tol:
            break
    return map(float, x)
    # raise RuntimeError("Failed to converge")
