"""Best-only checkpoint helpers with storage discipline.

Storage rules (100 GB quota):
- Save best.pt only. Never write last.pt.
- Never save mBART or DINOv2 base weights; only trainable state deltas.
- Atomic write via temporary file plus os.replace.

Public API (to implement):
- def save_best(path: Path, payload: dict) -> None
- def load_best(path: Path) -> dict
- def trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]
    Returns only parameters with requires_grad=True (skips frozen base weights).
"""

raise NotImplementedError("pgat_length.training.checkpoint: implement in step 04")
