import torch
import random
import numpy as np
import torch.nn as nn
from typing import Tuple
import torch.nn.init as init
from torch.nn import Parameter
from mycode.s3r import S3R
from mycode.SL import StructuralNetwork

class DGTEN(torch.nn.Module):
    """Historical class name for the GDTE model used by mainZ.py.

    GDTE is inspired by DGTEN, but this implementation is deterministic:
    uncertainty modeling is not used, and structural mappings are provided by
    SpectralGatedMapping in mycode.SL rather than Gaussian mapping.
    """
    def __init__(self, device, args, X, num_labels, index_list):
        super(DGTEN, self).__init__()
        self.args = args
        self.set_seed(self.args.seed)
        self.device = device
        self.X = X
        self.nodes = X.shape[0]
        self.num_labels = num_labels
        self.index_list = index_list
        self.dropout = self.args.dropout
        self.input_dim = self.args.layers[-1]
        self.num_heads = self.args.attention_head
        self.timeslots = self.args.train_time_slots

        self._build_model()
        self._initialize_loss_functions()

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def _build_model(self) -> None:
        self.structural_layer = StructuralNetwork(
            self.device,
            self.args,
            self.X,
            self.num_labels,
        ).to(self.device)

        self.s3r = S3R(
            input_dim=self.input_dim,
            n_heads=self.num_heads,
            max_seq_len=self.args.train_time_slots,
            dropout=self.dropout,
        ).to(self.device)

    def _initialize_loss_functions(self) -> None:
        self.mse_loss_func = nn.MSELoss()
        self.regression_weights = Parameter(
            torch.Tensor(self.args.layers[-1]*2, self.num_labels)
        )
        init.xavier_normal_(self.regression_weights)

    def _calculate_prediction_loss(self, node_embeddings, train_edges, target) -> torch.Tensor:
        start_node, end_node = node_embeddings[train_edges[0], :], node_embeddings[train_edges[1], :]
        features = torch.cat((start_node, end_node), 1)
        predictions = torch.mm(features, self.regression_weights)
        counts = np.bincount(target.cpu().clamp(0, 1))
        if len(counts) < 2:
            counts = np.array([1, 1])
        weight = torch.FloatTensor(1.0 / (counts + 1) * features.size(0)).to(self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight).to(self.device)
        return criterion(predictions, target)

    def embedding_consistency_loss(self, mu_clean: torch.Tensor, mu_adv: torch.Tensor) -> torch.Tensor:
        return self.mse_loss_func(mu_adv, mu_clean.detach())

    def _process_structural_layer(self, train_edges, train_labels, edge_weights=None) -> torch.Tensor:
        structural_out = []
        index0 = 0
        for i in range(self.args.train_time_slots):
            end = self.index_list[i] + 1
            lim = slice(index0, end)
            edges = train_edges[:, lim]
            labels = train_labels[lim, :]
            weights = edge_weights[lim] if edge_weights is not None else None
            current = self.structural_layer(edges, labels, weights)
            structural_out.append(current)
            index0 = end
        x_sequence = torch.stack(structural_out).permute(1, 0, 2)
        return x_sequence

    def forward(self, train_edges, y, train_labels, adversarial=False, edge_weights=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels_for_gnn = train_labels

        if adversarial and self.training:
            flip_ratio = float(getattr(self.args, "adv_flip_ratio", 0.03))
            flip_mask = torch.rand(train_labels.shape[0], device=self.device) < flip_ratio
            if flip_mask.any():
                labels_for_gnn = train_labels.clone()
                labels_for_gnn[flip_mask] = labels_for_gnn[flip_mask].flip(dims=[1])

        mu = self._process_structural_layer(train_edges, labels_for_gnn, edge_weights=edge_weights)
        temporal_all = self.s3r(mu)
        temporal_out = temporal_all[:, self.args.train_time_slots-1, :].squeeze()

        prediction_loss = self._calculate_prediction_loss(temporal_out, train_edges, y)
        return prediction_loss, temporal_out, mu
