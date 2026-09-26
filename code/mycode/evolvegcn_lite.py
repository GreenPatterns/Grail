"""
EvolveGCNLite — a THIRD, independent snapshot-window trust GNN.

A different family again from GDTE (spectral-gated RFF + S3R attention) and from
SignedSnapshotGNN (signed mean pools + GRU temporal head). Here the temporal
mechanism is EvolveGCN-style WEIGHT EVOLUTION: a single GCN weight matrix is
carried as the hidden state of a GRU and updated from snapshot to snapshot, while
node states propagate through a signed, degree-normalised graph convolution.

Crucially it shares only the property the placement hazard needs: it processes a
prefix ``range(train_time_slots)`` of a boundary list that may be longer, so an
edge appended past the prefix is never embedded (Proposition 1). Drop-in for
ReputationAttackOracle (same attributes/methods the oracle reads).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


class _Identity(nn.Module):
    """Temporal head is folded into weight evolution, so s3r is a pass-through."""
    def forward(self, mu):
        return mu


class EvolveGCNLite(nn.Module):
    def __init__(self, device, args, X, num_labels, index_list, hidden=16):
        super().__init__()
        self.args = args
        self.device = device
        self.X = (X if torch.is_tensor(X) else torch.as_tensor(X, dtype=torch.float)).float()
        self.nodes = self.X.shape[0]
        self.num_labels = num_labels
        self.index_list = index_list
        self.input_dim = hidden
        self.hidden = hidden

        raw = self.X.shape[1]
        self.proj = nn.Linear(raw, hidden)
        self.W_self = nn.Linear(hidden, hidden, bias=False)
        # GRU that evolves the graph-conv weight across snapshots, as a residual
        # around a learnable base so the (label-carrying) graph term stays
        # informative rather than collapsing.
        self.evolve = nn.GRUCell(hidden, hidden * hidden)
        self.W_base = nn.Parameter(torch.empty(hidden, hidden))
        nn.init.xavier_uniform_(self.W_base)
        self.s3r = _Identity()
        self.regression_weights = nn.Parameter(torch.empty(hidden * 2, num_labels))
        nn.init.xavier_normal_(self.regression_weights)
        self.to(device)
        self.X = self.X.to(device)

    def _snapshot_conv(self, H, edges, labels, W):
        """Signed, degree-normalised graph conv into destinations with weight W."""
        if edges.numel() == 0:
            return F.relu(self.W_self(H))
        row, col = edges[0], edges[1]
        # soft signed weight: trust [1,0] -> +1, distrust [0,1] -> -1,
        # neutral [0.5,0.5] -> 0, so a neutral edge contributes no signal
        # (continuous in the label; matches the sign-test semantics).
        sign = labels[:, 0] - labels[:, 1]
        msg = sign[:, None] * H[row]
        agg = scatter_add(msg, col, dim=0, dim_size=self.nodes)
        deg = scatter_add(torch.ones_like(sign), col, dim=0, dim_size=self.nodes).clamp(min=1.0)
        agg = agg / deg[:, None]
        return F.relu(agg @ W + self.W_self(H))

    def _process_structural_layer(self, train_edges, train_labels, edge_weights=None):
        H = F.relu(self.proj(self.X))
        w_flat = torch.zeros(1, self.hidden * self.hidden, device=self.X.device)
        out, index0 = [], 0
        for i in range(self.args.train_time_slots):
            end = self.index_list[i] + 1
            lim = slice(index0, end)
            w_flat = self.evolve(H.mean(0, keepdim=True), w_flat)       # evolve delta
            W = self.W_base + 0.1 * torch.tanh(w_flat).view(self.hidden, self.hidden)
            H = self._snapshot_conv(H, train_edges[:, lim], train_labels[lim, :], W)
            out.append(H)
            index0 = end
        return torch.stack(out).permute(1, 0, 2)        # [N, T, hidden]

    def forward(self, train_edges, y, train_labels, **kw):
        mu = self._process_structural_layer(train_edges, train_labels)
        temporal_out = self.s3r(mu)[:, self.args.train_time_slots - 1, :].squeeze()
        feats = torch.cat((temporal_out[train_edges[0]], temporal_out[train_edges[1]]), dim=1)
        logits = feats @ self.regression_weights
        counts = torch.bincount(y, minlength=self.num_labels).float()
        w = (counts.sum() / (counts + 1.0)).to(self.device)
        return F.cross_entropy(logits, y, weight=w), temporal_out, mu
