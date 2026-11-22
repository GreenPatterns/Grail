"""Reputation aggregation over a target's incoming raters' predicted trust P(trust | u, v).

  mean    : Eq. 1 of the paper, the undefended default
  median  : robust to a minority of outlying raters
  trimmed : mean after dropping the lowest and highest `trim` fraction of raters;
            sets too small to trim are left intact

Used by ReputationAttackOracle._reputation_score (rep_mode 'median' / 'trimmed') and by
the robust-functional baseline (experiments/run_robust_functional.py).
"""
import torch


def aggregate_reputation(p_trust: torch.Tensor, mode: str = "mean", trim: float = 0.1) -> torch.Tensor:
    if mode == "mean":
        return p_trust.mean()
    if mode == "median":
        s, _ = torch.sort(p_trust)
        n = s.numel()
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    if mode == "trimmed":
        s, _ = torch.sort(p_trust)
        n = s.numel()
        k = int(n * trim)
        if n - 2 * k < 1:
            k = 0
        return s[k:n - k].mean()
    raise ValueError(f"unknown reputation mode {mode!r}")
