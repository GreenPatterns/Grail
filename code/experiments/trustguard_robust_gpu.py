"""
GPU implementation of TrustGuard's robust coefficient (robust_aggregation.ro_coefficient of the
official release), used by run_trustguard_casestudy.py to run TrustGuard with its robust
aggregation ON.

Why it exists: the released function builds a dense N x N cosine-similarity matrix and several
dense/sparse copies on the CPU for every layer, snapshot, and direction (42 calls per forward
pass), returns a CPU tensor that cannot be multiplied with the CUDA tensors of the forward pass,
and uses the sparse `.A` attribute that SciPy >= 1.14 removed. `ro_coefficient_reference` is the
released function with only `.A` replaced by `.toarray()`; `ro_coefficient_gpu` computes the same
quantity on the device of the embeddings:

  sim_uv = 1 + cos(x_u, x_v) for each distinct edge (u, v); sim < prune -> 0; self-pairs -> 0;
  row-normalise over u's distinct edges, multiply by u's count of non-zero entries, add the
  release's self-loop diagonal (ones on the first L nodes, L = number of self-loop edges, as the
  release writes `ones[i] = 1` for i < L), row-normalise again, read the value of every edge.

`python trustguard_robust_gpu.py` checks the two against each other on random inputs with
duplicate edges and self-loops (the check on real TrustGuard inputs is in the case-study script).
"""
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def ro_coefficient_reference(node_embedding, prune, row, col):
    """The released ro_coefficient, verbatim except `.A` -> `.toarray()` (SciPy >= 1.14)."""
    node_embedding_copy = node_embedding.cpu().data.numpy()
    sim_matrix = cosine_similarity(node_embedding_copy)
    sim_matrix = 1 + sim_matrix
    sim = sim_matrix[row, col]
    sim[sim < prune] = 0
    loop_list = [row[i] for i in range(len(row)) if row[i] == col[i]]
    att_dense = lil_matrix((node_embedding_copy.shape[0], node_embedding_copy.shape[0]), dtype=np.float32)
    att_dense[row, col] = sim
    att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format='lil')
    att_dense_norm = normalize(att_dense, axis=1, norm='l1')
    degree = (att_dense_norm != 0).sum(1)
    att_dense_tran = lil_matrix(att_dense_norm.toarray() * np.asarray(degree))
    ones = np.zeros((len(degree),))
    for i in range(len(loop_list)):
        ones[i] = 1
    self_loop = sp.diags(ones, offsets=0, format='lil')
    att_final = att_dense_tran + self_loop
    att_final_norm = normalize(att_final, axis=1, norm='l1')
    att_real = att_final_norm[row, col]
    att_real_final = np.reshape(np.asarray(att_real), (len(row), 1))
    return torch.from_numpy(att_real_final).type(torch.float)


def ro_coefficient_gpu(node_embedding, prune, row, col):
    """Same value as ro_coefficient_reference, computed on node_embedding's device (no gradient)."""
    x = node_embedding.detach()
    dev, n = x.device, x.size(0)
    row = torch.as_tensor(np.asarray(row), device=dev, dtype=torch.long)
    col = torch.as_tensor(np.asarray(col), device=dev, dtype=torch.long)
    norms = x.norm(dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)   # sklearn leaves zero rows zero
    xn = x / norms
    uniq, inv = torch.unique(row * n + col, return_inverse=True)     # one matrix entry per pair
    ur, uc = uniq // n, uniq % n
    sim = 1 + (xn[ur] * xn[uc]).sum(1)
    zero = torch.zeros_like(sim)
    sim = torch.where(sim < prune, zero, sim)
    sim = torch.where(ur == uc, zero, sim).double()                  # diagonal removed
    s = torch.zeros(n, dtype=torch.float64, device=dev).index_add_(0, ur, sim)
    deg = torch.zeros(n, dtype=torch.float64, device=dev).index_add_(0, ur, (sim != 0).double())
    n_loops = int((row == col).sum())
    selfl = (torch.arange(n, device=dev) < n_loops).double()
    s_r, deg_r = s[ur], deg[ur]
    tran = torch.where(s_r > 0, sim / s_r.clamp_min(1e-300) * deg_r, torch.zeros_like(sim))
    denom = deg + selfl
    val = torch.where(denom[ur] > 0, tran / denom[ur].clamp_min(1e-300), torch.zeros_like(tran))
    diag = torch.where(denom > 0, selfl / denom.clamp_min(1e-300), torch.zeros_like(denom))
    val = torch.where(ur == uc, diag[ur], val)
    return val[inv].float().unsqueeze(1)


def max_abs_diff(x, prune, row, col):
    """Largest |gpu - reference| over all edges, and the number of edges whose value differs by >1e-5."""
    ref = ro_coefficient_reference(x.cpu(), prune, row, col).numpy().ravel()
    gpu = ro_coefficient_gpu(x, prune, row, col).cpu().numpy().ravel()
    d = np.abs(ref - gpu)
    return float(d.max()) if len(d) else 0.0, int((d > 1e-5).sum())


if __name__ == '__main__':
    rng = np.random.RandomState(0)
    for trial in range(5):
        n, e = 300, 2000
        x = torch.from_numpy(rng.normal(size=(n, 8)).astype(np.float32))
        x[:3] = 0                                                     # zero rows
        row, col = rng.randint(0, n, e), rng.randint(0, n, e)
        row[:20], col[:20] = row[20:40], col[20:40]                   # duplicate edges
        col[40:45] = row[40:45]                                       # self-loops
        dmax, nbad = max_abs_diff(x, 1.4 if trial % 2 else 1.0, row, col)
        assert nbad == 0 and dmax < 1e-6, (trial, dmax, nbad)
    print('ro_coefficient_gpu matches the reference on 5 random graphs')
