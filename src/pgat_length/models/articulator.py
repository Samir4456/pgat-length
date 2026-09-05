"""Biased articulator attention (8 learned queries).

Responsibility:
- Attend from 8 learned articulator queries into the K_temporal temporal tokens.
- Bias attention logits with pose confidence and motion magnitude.
- Query groups: left hand, right hand, mouth, upper body / global pose.

Public API (to implement):
- class BiasedArticulatorAttention(nn.Module):
    forward(temporal_tokens, valid_mask, pose_confidence, pose_motion)
        -> articulator_tokens [B, 8, hidden_dim]
"""

raise NotImplementedError("pgat_length.models.articulator: implement in step 02")
