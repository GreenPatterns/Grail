import math
import torch
import random
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean



def uniform(size, tensor):
    if tensor is not None:
        if not isinstance(tensor, torch.Tensor):
            return
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)

class SpectralGatedMapping(nn.Module):
    def __init__(self, input_dim, output_dim, rff_dim=None):
        super().__init__()
        if rff_dim is None:
            rff_dim = output_dim // 2
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rff_dim = rff_dim
        self.W = nn.Parameter(torch.randn(input_dim, rff_dim))
        self.b = nn.Parameter(torch.rand(rff_dim) * 2 * math.pi)
        hidden = output_dim
        self.W_gate = nn.Linear(2 * rff_dim, hidden, bias=False)
        self.W_value = nn.Linear(2 * rff_dim, hidden, bias=False)
        self.head = nn.Linear(hidden, output_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.W, 0, 1)
        nn.init.uniform_(self.b, 0, 2 * math.pi)
        nn.init.xavier_uniform_(self.W_gate.weight)
        nn.init.xavier_uniform_(self.W_value.weight)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
    def forward(self, x):
        proj = x @ self.W + self.b
        rff = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        features = self.W_value(rff) * F.silu(self.W_gate(rff))
        return self.head(features)

class Convolution(nn.Module):
    def __init__(self, in_channels_d, out_channels, num_labels, prune, device, robust_aggr=False, bias=True):
        super().__init__()
        self.in_channels_d = in_channels_d
        self.out_channels = out_channels
        self.prune = prune
        self.prune_js = 0.4 if prune > 1.0 else prune
        self.device = device
        self.robust_aggr = robust_aggr
        self.num_labels = num_labels
        self.concatenated_dim = self.in_channels_d * 4
        self.transformation_gaussian = SpectralGatedMapping(input_dim=self.num_labels, output_dim=self.in_channels_d)
        self.weight = Parameter(torch.Tensor(out_channels, self.concatenated_dim))
        self.alpha = 0.4
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.weight.size(0), self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def outgoing_edges(self, feature_mu, label_mu, edge_index, external_trust=None):
        row, col = edge_index
        num_nodes = feature_mu.size(0)

        if not self.robust_aggr:
            opinion_mu = scatter_mean(label_mu, row, dim=0, dim_size=num_nodes)
            out_mu = scatter_mean(feature_mu[col], row, dim=0, dim_size=num_nodes)
        else:
            raise NotImplementedError(
                "robust_aggr=True requires mycode.robust module "
                "(ro_coefficient_dense_ensemble_complete). "
                "Set robust_aggr=False or provide the robust module."
            )
        return out_mu, opinion_mu

    def incoming_edges(self, feature_mu, label_mu, edge_index, external_trust=None):
        row, col = edge_index
        num_nodes = feature_mu.size(0)

        if not self.robust_aggr:
            opinion_mu = scatter_mean(label_mu, col, dim=0, dim_size=num_nodes)
            out_mu = scatter_mean(feature_mu[row], col, dim=0, dim_size=num_nodes)
        else:
            raise NotImplementedError(
                "robust_aggr=True requires mycode.robust module "
                "(ro_coefficient_dense_ensemble_complete). "
                "Set robust_aggr=False or provide the robust module."
            )
        return out_mu, opinion_mu

    def forward(self, feature_mu, edge_index, edge_label, external_trust=None):
        if edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_label.size(1) != self.num_labels:
            raise ValueError(f"edge_label must have shape [num_edges, {self.num_labels}]")
        edge_index = edge_index.to(feature_mu.device)
        edge_label = edge_label.to(feature_mu.device)
        label_mu = self.transformation_gaussian(edge_label)

        out_mu, opinion_mu = self.outgoing_edges(
            feature_mu, label_mu, edge_index, external_trust
        )
        inn_mu, inn_opinion_mu = self.incoming_edges(
            feature_mu, label_mu, edge_index, external_trust
        )

        mu_concat = torch.cat((out_mu, opinion_mu, inn_mu, inn_opinion_mu), dim=1)
        mu_linear = F.linear(mu_concat, self.weight, self.bias)
        mu_activated = F.relu(mu_linear)
        return mu_activated

class SpatialConvolutionBase(Convolution):
    pass
class SpatialConvolutionLayer(Convolution):
    pass

class StructuralNetwork(nn.Module):
    def __init__(self, device, args, X, num_labels, prune=0.4):
        super().__init__()
        self.args = args
        self.setSeed(self.args.seed)
        self.device = device
        self.X_raw = X
        self.dropout = args.dropout
        self.robust_aggr = getattr(args, "robust_aggr", False)
        self.num_labels = num_labels
        self.prune = prune
        self.raw_input_dim = self.X_raw.size(1)
        self.target_gaussian_dim = args.layers[0] if args.layers else 64

        self.projection = None
        current_feature_dim = self.raw_input_dim
        if self.raw_input_dim != self.target_gaussian_dim:
            self.projection = nn.Linear(self.raw_input_dim, self.target_gaussian_dim).to(device)
            uniform(self.raw_input_dim, self.projection.weight)
            if self.projection.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.projection.weight)
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.projection.bias, -bound, bound)
            current_feature_dim = self.target_gaussian_dim

        self.initial_node_mapper = SpectralGatedMapping(
            input_dim=current_feature_dim,
            output_dim=self.target_gaussian_dim
        ).to(device)
        self.setup_layers()

    def setSeed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def setup_layers(self):
        if self.X_raw.shape[0] == 0:
            raise ValueError("Input features X cannot be empty")

        self.neurons = self.args.layers
        self.num_layers = len(self.neurons)
        if self.num_layers == 0:
            raise ValueError("args.layers cannot be empty")

        self.conv_layers = nn.ModuleList()
        conv_base = SpatialConvolutionBase(
            in_channels_d=self.target_gaussian_dim,
            out_channels=self.neurons[0],
            num_labels=self.num_labels,
            prune=self.prune,
            device=self.device,
            robust_aggr=self.robust_aggr,
        ).to(self.device)
        self.conv_layers.append(conv_base)

        current_gaussian_dim = self.neurons[0]
        for i in range(1, self.num_layers):
            conv_layer = SpatialConvolutionLayer(
                in_channels_d=current_gaussian_dim,
                out_channels=self.neurons[i],
                num_labels=self.num_labels,
                prune=self.prune,
                device=self.device,
                robust_aggr=self.robust_aggr,
            ).to(self.device)
            self.conv_layers.append(conv_layer)
            current_gaussian_dim = self.neurons[i]

    def forward(self, edge_index, edge_label, external_trust=None):
        x_feat = self.X_raw.to(self.device)
        if self.projection is not None:
            x_feat = self.projection(x_feat)
        mu = self.initial_node_mapper(x_feat)
        for i, layer in enumerate(self.conv_layers):
            mu_dropped = F.dropout(mu, self.dropout, training=self.training)
            mu = layer(mu_dropped, edge_index, edge_label, external_trust)
        return mu
    def __repr__(self):
        return f"{self.__class__.__name__}(num_layers={self.num_layers}, neurons={self.neurons})"
