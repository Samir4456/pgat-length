"""PGAT-v2 tokenizer with VARIABLE K_temporal per sample.

Adopts the PGAT-v1 six-source gated fusion and the temporal Transformer,
but the temporal length K is per-sample (not fixed to 16).

Inputs per sample (shapes vary by K = K_temporal):
- spatial_features:  [K, 4, 768]  (global + 3 articulators)
- spatial_valid:     [K, 4]
- motion_features:   [8, 768]     (interpolated to K by segment centers)
- pose_descriptor:   [K, 64]
- pose_confidence:   [K, 4]
- pose_motion:       [K, 4]
- segment_valid:     [K]

Outputs (batched with padding, so effective K in a batch is max_K):
- temporal_tokens:   [B, max_K, hidden_dim]
- valid_mask:        [B, max_K]  (True for real segments, False for padding)

Public API (to implement):
- class PgatVariableTokenizer(nn.Module):
    forward(spatial, spatial_valid, motion, motion_centers, pose_descriptor,
            pose_confidence, pose_motion, segment_valid) -> (temporal_tokens, valid_mask)
"""

raise NotImplementedError("pgat_length.models.tokenizer: implement in step 02")
