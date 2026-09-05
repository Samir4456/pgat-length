"""Global summary attention (4 learned queries).

Responsibility:
- Standard multi-head attention: 4 learned global queries attend into
  the K_temporal temporal tokens with the valid_mask as key padding mask.
- Residual + LayerNorm + feed-forward (same shape).

Public API (to implement):
- class GlobalSummaryAttention(nn.Module):
    forward(temporal_tokens, valid_mask) -> global_tokens [B, 4, hidden_dim]
"""

raise NotImplementedError("pgat_length.models.global_summary: implement in step 02")
