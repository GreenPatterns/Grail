"""Exact fast scorer for distrust edges spliced into the last processed snapshot.

GDTE's structural layer embeds each snapshot from that snapshot's edges alone
(DGTEN._process_structural_layer; StructuralNetwork keeps no state across snapshots), and the
temporal layer (S3R) reads the stacked sequence. An edge spliced at the end of snapshot
s_{T-1}, the processed placement of the paper, therefore changes only that snapshot's slice
of the sequence. S6Scorer caches the clean slices s_0..s_{T-2} once per model and, per
injection, recomputes the s_{T-1} slice and the temporal layer. The result equals the
oracle's full forward pass up to float nondeterminism (tests/test_fast_s6.py). S3R also
treats every node's time sequence independently (its attention, top-k and state-space scan
run along the time axis within a row), so after an injection only the rows that enter the
target's reputation (the target, its raters and the injected sources) are recomputed.

Reputation is Eq. 1 of the paper, mean P(trust | u, v) over v's incoming raters: the total
effect averages all raters including the injected edges, the propagation-only effect only
the pre-existing ones, both with the attacked embeddings.
"""
import torch
import torch.nn.functional as F


class S6Scorer:
    def __init__(self, oracle, edges, labels):
        self.model = oracle.model
        assert not self.model.training, "the oracle's model must be in eval mode"
        self.T = self.model.args.train_time_slots
        idx = list(self.model.index_list)
        self.lo, self.hi = idx[self.T - 2] + 1, idx[self.T - 1] + 1
        assert edges.shape[1] == self.hi, "edges must be exactly the processed window"
        self.edges, self.labels, self.device = edges, labels, edges.device
        with torch.no_grad():
            self.mu = self.model._process_structural_layer(edges, labels)
            self.z0 = self.model.s3r(self.mu)[:, self.T - 1, :]
        self.dst = edges[1]

    def _p_trust(self, z, src, target):
        feats = torch.cat((z[src], z[target].unsqueeze(0).expand(len(src), -1)), dim=1)
        return F.softmax(feats @ self.model.regression_weights, dim=1)[:, 0]

    def clean_vector(self, target):
        """P(trust | u, target) for every pre-existing rater u, clean embeddings."""
        cols = (self.dst == target).nonzero(as_tuple=True)[0]
        with torch.no_grad():
            return self._p_trust(self.z0, self.edges[0, cols], target)

    def clean(self, target):
        return self.clean_vector(target).mean().item()

    def attacked(self, target, srcs, label=(0.0, 1.0)):
        """(total-effect R, propagation-only R) after splicing edges srcs -> target, each with
        one-hot label `label` (distrust by default), at the end of s_{T-1}."""
        p_pre, p_inj = self.attacked_vectors(target, srcs, label)
        r_full = torch.cat([p_pre, p_inj]).mean().item()
        r_pre = p_pre.mean().item() if len(p_pre) else float('nan')
        return r_full, r_pre

    def attacked_vectors(self, target, srcs, label=(0.0, 1.0)):
        """(P(trust) of the pre-existing raters, P(trust) of the injected edges), both from the
        attacked embeddings; any robust aggregate of the raters can be taken from them."""
        B = len(srcs)
        ne = torch.tensor([list(srcs), [target] * B], dtype=self.edges.dtype, device=self.device)
        nl = torch.tensor([list(label)] * B, dtype=self.labels.dtype, device=self.device)
        s6_e = torch.cat([self.edges[:, self.lo:self.hi], ne], dim=1)
        s6_l = torch.cat([self.labels[self.lo:self.hi], nl], dim=0)
        with torch.no_grad():
            mu6 = self.model.structural_layer(s6_e, s6_l)
            pre = (self.dst == target).nonzero(as_tuple=True)[0]
            src_pre = self.edges[0, pre]
            src_all = torch.cat([src_pre, ne[0]])
            rows, inv = torch.unique(torch.cat([torch.tensor([target], device=self.device), src_all]),
                                     return_inverse=True)
            mu_rows = self.mu[rows].clone()
            mu_rows[:, self.T - 1, :] = mu6[rows]
            z_rows = self.model.s3r(mu_rows)[:, self.T - 1, :]
            z_t, z_all, n_pre = z_rows[inv[0]], z_rows[inv[1:]], len(src_pre)

            feats = torch.cat((z_all, z_t.unsqueeze(0).expand(len(z_all), -1)), dim=1)
            p = F.softmax(feats @ self.model.regression_weights, dim=1)[:, 0]
        return p[:n_pre], p[n_pre:]

    def best_single(self, target, pool):
        """The candidate whose single distrust edge lowers the total-effect R most (the paper's
        exact per-candidate counterfactual), with its (R_total, R_prop)."""
        best = None
        for s in pool:
            rf, rp = self.attacked(target, [s])
            if best is None or rf < best[1]:
                best = (s, rf, rp)
        return best
