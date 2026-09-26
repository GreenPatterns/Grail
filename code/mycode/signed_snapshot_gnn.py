"""
SignedSnapshotGNN — an INDEPENDENT second snapshot-window trust GNN.

Purpose (reviewer Q4): demonstrate that the silent-no-op placement hazard is a
property of the prefix-snapshot *pattern*, not of GDTE specifically. This model
is deliberately a different architecture family from GDTE:

    GDTE structural layer : spectral-gated random-Fourier-feature mapping (SL.py)
    GDTE temporal head    : S3R self-attention (s3r.py)

    This model structural  : SGCN/SDGNN-style SIGNED mean aggregation
                             (separate positive / negative neighbour pools)
    This model temporal    : a GRU over the snapshot sequence

What it shares with GDTE — and the only thing the hazard needs — is the
prefix-snapshot processing structure: ``_process_structural_layer`` iterates
``range(train_time_slots)`` over disjoint snapshot slices of a boundary list
that has K >= train_time_slots entries. An edge appended past the processed
prefix is therefore never embedded, so its label cannot reach the encoder.

It exposes the same attributes/methods the ReputationAttackOracle reads
(``nodes, input_dim, num_labels, args, index_list, regression_weights,
_process_structural_layer, s3r``), so it is a drop-in for the existing
diagnostics with no oracle changes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


class _GRUTemporal(nn.Module):
    """Temporal head: GRU over the snapshot sequence (vs GDTE's S3R attention)."""
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, embed_dim, batch_first=True)

    def forward(self, mu):                      # mu: [N, T, input_dim]
        out, _ = self.gru(mu)
        return out                              # [N, T, embed_dim]


class SignedSnapshotGNN(nn.Module):
    def __init__(self, device, args, X, num_labels, index_list, hidden=32, n_layers=2):
        super().__init__()
        self.args = args
        self.device = device
        self.X = X if torch.is_tensor(X) else torch.as_tensor(X, dtype=torch.float)
        self.X = self.X.float()
        self.nodes = self.X.shape[0]
        self.num_labels = num_labels
        self.index_list = index_list
        self.input_dim = hidden                 # embedding dim seen by the head
        self.n_layers = n_layers

        raw = self.X.shape[1]
        self.proj = nn.Linear(raw, hidden)
        # signed convolution: self / positive-neighbour / negative-neighbour maps
        self.W_self = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.W_pos = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.W_neg = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.s3r = _GRUTemporal(hidden, hidden)
        self.regression_weights = nn.Parameter(torch.empty(hidden * 2, num_labels))
        nn.init.xavier_normal_(self.regression_weights)
        self.to(device)
        self.X = self.X.to(device)

    def _snapshot_embed(self, edges, labels):
        """One snapshot: SIGNED mean aggregation of source features into dests,
        with separate positive/negative neighbour pools (SGCN/SDGNN style).

        Pools are weighted by the soft label (column 0 = trust, column 1 =
        distrust), so a trust edge contributes to the positive pool, a distrust
        edge to the negative pool, and a neutral [0.5,0.5] edge to both equally.
        This is continuous in the label, unlike GDTE's RFF label mapping.
        """
        x = F.relu(self.proj(self.X))
        if edges.numel() == 0:
            for li in range(self.n_layers):
                x = F.relu(self.W_self[li](x))
            return x
        row, col = edges[0], edges[1]           # src -> dst (reputation = incoming)
        eps = 1e-9
        w_pos = labels[:, 0].clamp(min=0)       # trust weight per edge
        w_neg = labels[:, 1].clamp(min=0)       # distrust weight per edge
        denom_pos = scatter_add(w_pos, col, dim=0, dim_size=self.nodes).clamp(min=eps)
        denom_neg = scatter_add(w_neg, col, dim=0, dim_size=self.nodes).clamp(min=eps)
        for li in range(self.n_layers):
            msg = x[row]
            pos = scatter_add(msg * w_pos[:, None], col, dim=0, dim_size=self.nodes) / denom_pos[:, None]
            neg = scatter_add(msg * w_neg[:, None], col, dim=0, dim_size=self.nodes) / denom_neg[:, None]
            x = F.relu(self.W_self[li](x) + self.W_pos[li](pos) + self.W_neg[li](neg))
        return x                                # [N, hidden]

    def _process_structural_layer(self, train_edges, train_labels, edge_weights=None):
        """Prefix-snapshot processing — identical control structure to GDTE."""
        out, index0 = [], 0
        for i in range(self.args.train_time_slots):
            end = self.index_list[i] + 1
            lim = slice(index0, end)
            out.append(self._snapshot_embed(train_edges[:, lim], train_labels[lim, :]))
            index0 = end
        return torch.stack(out).permute(1, 0, 2)        # [N, T, hidden]

    def forward(self, train_edges, y, train_labels, **kw):
        mu = self._process_structural_layer(train_edges, train_labels)
        temporal_all = self.s3r(mu)
        temporal_out = temporal_all[:, self.args.train_time_slots - 1, :].squeeze()
        feats = torch.cat((temporal_out[train_edges[0]], temporal_out[train_edges[1]]), dim=1)
        logits = feats @ self.regression_weights
        # inverse-frequency class weights (trust dominates) to avoid collapse
        counts = torch.bincount(y, minlength=self.num_labels).float()
        w = (counts.sum() / (counts + 1.0)).to(self.device)
        loss = F.cross_entropy(logits, y, weight=w)
        return loss, temporal_out, mu
