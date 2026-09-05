"""Video-text alignment head for stage 1 contrastive pretraining.

Responsibility:
- Wrap the PGAT variable tokenizer + articulator + global summary.
- Pool visual tokens into a single video embedding (mean over 4 global tokens).
- Encode text via mBART encoder (frozen at this stage) + mean-pool + project.
- Return L2-normalized video and text embeddings for contrastive loss.

Public API (to implement):
- class PgatAlignmentModel(nn.Module):
    encode_video(batch) -> video_embedding [B, 768]
    encode_text(input_ids, attention_mask) -> text_embedding [B, 768]
    forward(batch) -> (video_embedding, text_embedding)
- class InfoNceHardNegatives(nn.Module):
    forward(video_emb, text_emb, temperature) -> loss
"""

raise NotImplementedError("pgat_length.models.alignment: implement in step 04")
