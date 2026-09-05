"""Regression tests for variable-length prefix shapes and masking.

Tests to implement:
- test_k_temporal_bounds:
    F=32   -> K=12 (min clamp)
    F=64   -> K=8  -> clamped to 12
    F=96   -> K=12
    F=200  -> K=25
    F=400  -> K=40 (max clamp)
- test_prefix_total_shape:
    K + 8 articulator + 4 global = prefix_length
- test_collator_padding_mask:
    Batch of K in {12, 20, 40} -> padded to max_K=40; attention mask valid on real, 0 on pad.
- test_projection_output_dim:
    PGAT [B, T, 512] -> mBART-compatible [B, T, 1024].
"""

import pytest


@pytest.mark.skip(reason="TODO: implement in step 03")
def test_k_temporal_bounds() -> None:
    ...


@pytest.mark.skip(reason="TODO: implement in step 03")
def test_prefix_total_shape() -> None:
    ...


@pytest.mark.skip(reason="TODO: implement in step 03")
def test_collator_padding_mask() -> None:
    ...


@pytest.mark.skip(reason="TODO: implement in step 03")
def test_projection_output_dim() -> None:
    ...
