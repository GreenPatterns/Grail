import math
import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

def xavier(t: torch.Tensor) -> torch.Tensor:
    nn.init.xavier_uniform_(t)  # Fixed: in-place version
    return t

def _vectorized_ssm_scan(dA: torch.Tensor, dBu: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
    """SSM scan with optional chunking along the B (node) dimension.

    The recurrence h_t = h_{t-1} * dA_t + dBu_t operates independently per node,
    so chunking along B is mathematically equivalent to processing all nodes in
    one shot. The only constraint is to keep dA/dBu contiguous per chunk and to
    accumulate outputs in the original order.

    chunk_size=0 or chunk_size>=B falls back to the original full-batch path.
    Override the default via the S3R_SSM_CHUNK_SIZE env var (useful for tests
    that want to force the chunked path even when B is small).
    """
    import os
    env_chunk = os.environ.get("S3R_SSM_CHUNK_SIZE")
    if env_chunk is not None and env_chunk != "":
        try:
            chunk_size = int(env_chunk)
        except ValueError:
            pass

    B, L, D, N = dA.shape
    if chunk_size <= 0 or chunk_size >= B:
        dA_clamped = dA.clamp(min=1e-4, max=1.0)
        h = torch.zeros(B, D, N, device=dA.device, dtype=torch.float64)
        hs = torch.zeros(B, L, D, N, device=dA.device, dtype=torch.float64)
        for t in range(L):
            h = h * dA_clamped[:, t].double() + dBu[:, t].double()
            hs[:, t] = h
        return hs.float()

    dA_clamped = dA.clamp(min=1e-4, max=1.0)
    hs_chunks = []
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        dA_chunk = dA_clamped[start:end]
        dBu_chunk = dBu[start:end]
        b = end - start
        h = torch.zeros(b, D, N, device=dA.device, dtype=torch.float64)
        hs = torch.zeros(b, L, D, N, device=dA.device, dtype=torch.float64)
        for t in range(L):
            h = h * dA_chunk[:, t].double() + dBu_chunk[:, t].double()
            hs[:, t] = h
        hs_chunks.append(hs.float())
    return torch.cat(hs_chunks, dim=0)

class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        
        self.W_in  = nn.Parameter(xavier(torch.empty(d_model, self.d_inner * 2)))
        self.W_x   = nn.Parameter(xavier(torch.empty(self.d_inner, self.dt_rank + d_state * 2)))
        self.W_dt  = nn.Parameter(xavier(torch.empty(self.dt_rank, self.d_inner)))
        self.W_out = nn.Parameter(xavier(torch.empty(self.d_inner, d_model)))

        dt_bias_val = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        self.b_dt = nn.Parameter(torch.log(torch.expm1(dt_bias_val)))

        A = einops.repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            'n -> d n', d=self.d_inner
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(self.d_inner))

        self.d_conv = d_conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        xz   = x @ self.W_in
        u, z = xz.chunk(2, dim=-1)

        kernel_size = min(self.d_conv, L)
        weight      = self.conv1d.weight[:, :, :kernel_size]
        padding     = kernel_size - 1
        u_conv = F.conv1d(
            u.transpose(1, 2),
            weight,
            bias=None,
            stride=1,
            padding=padding,
            groups=self.d_inner
        ).transpose(1, 2)[:, :L, :]
        u = F.silu(u_conv)

        x_dbl = u @ self.W_x
        delta_r, B_ssm, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(delta_r @ self.W_dt + self.b_dt).clamp(min=1e-5, max=10.0)
        A  = -torch.exp(self.A_log.float())

        dA = torch.exp(
            einops.einsum(delta, A, 'b l d, d n -> b l d n')
        ).clamp(min=1e-6, max=1.0)

        dBu = einops.einsum(
            delta, B_ssm, 'b l d, b l n -> b l d n'
        ) * u.unsqueeze(-1)

        hs = _vectorized_ssm_scan(dA, dBu)

        y  = torch.sum(hs * C.unsqueeze(-2), dim=-1)
        y  = y * F.silu(z)
        y  = y + u * self.D
        return y @ self.W_out

class S3R(nn.Module):
    def __init__(
        self,
        input_dim:   int   = 32,
        n_heads:     int   = 8,
        max_seq_len: int   = 23,
        dropout:     float = 0.1,
        top_k_ratio: float = 0.1,
        base_window: int   = 3,
        residual:    bool  = True,
        seed:        int   = 42,
    ):
        super().__init__()
        if input_dim % n_heads != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by n_heads ({n_heads})")

        self.input_dim   = input_dim
        self.n_heads     = n_heads
        self.max_seq_len = max_seq_len
        self.head_dim    = input_dim // n_heads
        self.top_k_ratio = top_k_ratio
        self.base_window = base_window
        self.residual    = residual

        self.position_embeddings = nn.Parameter(torch.Tensor(max_seq_len, input_dim))
        nn.init.xavier_uniform_(self.position_embeddings)

        self.W_q        = nn.Parameter(xavier(torch.empty(input_dim, input_dim)))
        self.W_k        = nn.Parameter(xavier(torch.empty(input_dim, input_dim)))
        self.W_v        = nn.Parameter(xavier(torch.empty(input_dim, input_dim)))
        self.W_o        = nn.Parameter(xavier(torch.empty(input_dim, input_dim)))
        self.W_gate      = nn.Parameter(xavier(torch.empty(input_dim, 2 * input_dim)))
        self.W_ssm_bias = nn.Parameter(xavier(torch.empty(input_dim, n_heads)))

        self.attention_gain = nn.Parameter(torch.ones(1, 1, input_dim).clamp(max=1.0))
        self.ffn_gain       = nn.Parameter(torch.ones(1, 1, input_dim).clamp(max=1.0))
        self.attention_temp = nn.Parameter(
            torch.full((n_heads, 1, 1), 1.0 / math.sqrt(self.head_dim))
        )

        self.dropout   = nn.Dropout(dropout)
        self.ssm_block = SelectiveSSM(d_model=input_dim, d_state=16)
        self._init_sparse_patterns()

    def _init_sparse_patterns(self):
        patterns = []
        for head_idx in range(self.n_heads):
            pattern  = torch.zeros(self.max_seq_len, self.max_seq_len, dtype=torch.bool)
            dilation = 2 ** (head_idx % 4)
            for i in range(self.max_seq_len):
                pattern[i, i] = True
                for w in range(1, self.base_window + 1):
                    j = i - w * dilation
                    if j >= 0:
                        pattern[i, j] = True
                stride = max(1, self.max_seq_len // 4)
                for j in range(0, i + 1, stride):
                    pattern[i, j] = True 
                pattern[i, 0] = True
                exp_pos = 1
                while i - exp_pos >= 0:
                    pattern[i, i - exp_pos] = True
                    exp_pos *= 3
            patterns.append(pattern)
        self.register_buffer('sparse_pattern', torch.stack(patterns)) 

    def _compute_hybrid_sparse_attention(self, Q, K, V, static_patterns, ssm_context, seq_len):
        B = Q.shape[0]

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.attention_temp.unsqueeze(0)
        ssm_bias = (ssm_context @ self.W_ssm_bias).permute(0, 2, 1).unsqueeze(-1)
        scores   = scores + ssm_bias

        k = max(1, int(seq_len * self.top_k_ratio))
        _, topk_idx = torch.topk(scores, k, dim=-1)
        dyn_mask = torch.zeros_like(scores, dtype=torch.bool)
        dyn_mask.scatter_(-1, topk_idx, True)
        dyn_mask[:, :, :, 0] = True

        stat_mask   = static_patterns[:, :seq_len, :seq_len].unsqueeze(0).expand(
            B, self.n_heads, seq_len, seq_len)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(0)

        final_mask    = (stat_mask | dyn_mask) & causal_mask
        scores_masked = scores.masked_fill(~final_mask, float('-inf'))
        weights       = F.softmax(scores_masked, dim=-1)
        if self.training:
            weights = self.dropout(weights)

        output            = torch.matmul(weights, V)
        sparsity_per_head = 1.0 - final_mask.float().mean(dim=(0, 2, 3))
        return output, weights, sparsity_per_head 

    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape
        if L > self.max_seq_len:
            raise ValueError(f"Sequence length {L} exceeds maximum {self.max_seq_len}")

        pos_ids = torch.arange(L, device=x.device)
        h       = x + self.position_embeddings[pos_ids]
        ssm_ctx = self.ssm_block(h)

        Q = (h @ self.W_q).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = (h @ self.W_k).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = (h @ self.W_v).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out, attn_w, sparsity = self._compute_hybrid_sparse_attention(
            Q, K, V, self.sparse_pattern, ssm_ctx, L
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.input_dim)
        attn_out = attn_out @ self.W_o
        h        = h + attn_out * self.attention_gain

        glu = h @ self.W_gate
        val, gate = glu.chunk(2, dim=-1)
        ffn_out = val * torch.sigmoid(gate)
        if self.training:
            ffn_out = self.dropout(ffn_out)

        if self.residual:
            h = h + ffn_out * self.ffn_gain
        else:
            h = ffn_out

        return h