"""Non-GNN reputation baselines on signed weighted rating graphs.

fairness_goodness : Kumar, Spezzano, Subrahmanian & Faloutsos (ICDM 2016), "Edge Weight
    Prediction in Weighted Signed Networks", the algorithm released with Bitcoin-OTC/Alpha.
    With edge weights W(u,v) in [-1, 1], it alternates
        g(v) = (1/|in(v)|)  * sum_{u in in(v)}  f(u) * W(u,v)            (goodness, [-1, 1])
        f(u) = 1 - (1/(2|out(u)|)) * sum_{v in out(u)} |W(u,v) - g(v)|   (fairness, [0, 1])
    from f = g = 1 until convergence. A node with no incoming rating has goodness 0; a node
    that rates nobody keeps fairness 1. Goodness is the reputation; (g + 1) / 2 maps it to
    [0, 1] so the paper's 0.5 trust gate corresponds to g = 0.
    init=(f, g) warm-starts the iteration, e.g. from the clean graph's fixed point when a
    few edges are added; the iteration reaches the same fixed point in fewer steps.
"""
import numpy as np


def fairness_goodness(src, dst, w, n_nodes, max_iter=500, tol=1e-9, init=None):
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    indeg = np.bincount(dst, minlength=n_nodes).astype(np.float64)
    outdeg = np.bincount(src, minlength=n_nodes).astype(np.float64)
    has_in, has_out = indeg > 0, outdeg > 0
    if init is None:
        f, g = np.ones(n_nodes), np.ones(n_nodes)
    else:
        f, g = (np.array(x, dtype=np.float64, copy=True) for x in init)
    for _ in range(max_iter):
        num = np.bincount(dst, weights=f[src] * w, minlength=n_nodes)
        g_new = np.zeros(n_nodes)
        g_new[has_in] = num[has_in] / indeg[has_in]
        err = np.bincount(src, weights=np.abs(w - g_new[dst]), minlength=n_nodes)
        f_new = np.ones(n_nodes)
        f_new[has_out] = 1.0 - err[has_out] / (2.0 * outdeg[has_out])
        delta = max(np.abs(f_new - f).max(), np.abs(g_new - g).max())
        f, g = f_new, g_new
        if delta < tol:
            break
    return f, g
