"""Stage 2 translation training loop.

Warm-starts the PGAT encoder from alignment_best.pt (strict on the trainable
state dict portion) and fine-tunes:
    - PGAT tokenizer + articulator + global summary at pgat_lr
    - projection + mBART encoder + mBART decoder at mbart_lr

Uses label-smoothed cross-entropy, AdamW, cosine schedule with warmup,
bf16 mixed precision, gradient checkpointing on mBART, gradient clipping.
Early-stops on DEV loss. Saves translation_best.pt (trainable state only).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from pgat_length.data.collator import batch_to_device, collate_alignment_batch
from pgat_length.data.dataset import PhoenixCachedDataset
from pgat_length.models.tokenizer import EncoderConfig
from pgat_length.models.translation import PgatMbartTranslationModel, TranslationConfig
from pgat_length.training.checkpoint import load_best, save_best, trainable_state_dict


@dataclass(frozen=True)
class TranslationTrainingConfig:
    epochs: int
    micro_batch: int
    grad_accum: int
    encoder_lr: float
    decoder_lr: float
    pgat_lr: float
    weight_decay: float
    warmup_ratio: float
    label_smoothing: float
    max_target_tokens: int
    mixed_precision: str
    gradient_checkpointing: bool
    early_stop_patience: int
    seed: int


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_model(
    model_cfg: dict[str, Any],
    hf_cache: Path,
    label_smoothing: float,
) -> PgatMbartTranslationModel:
    encoder_cfg_raw = model_cfg["encoder"]
    encoder_cfg = EncoderConfig(
        spatial_dim=int(encoder_cfg_raw["spatial_dim"]),
        motion_dim=int(encoder_cfg_raw["motion_dim"]),
        pose_descriptor_dim=int(encoder_cfg_raw["pose_descriptor_dim"]),
        articulator_views=int(encoder_cfg_raw["articulator_views"]),
        hidden_dim=int(encoder_cfg_raw["hidden_dim"]),
        transformer_layers=int(encoder_cfg_raw["transformer_layers"]),
        transformer_heads=int(encoder_cfg_raw["transformer_heads"]),
        ffn_dim=int(encoder_cfg_raw["ffn_dim"]),
        dropout=float(encoder_cfg_raw["dropout"]),
    )
    translation_cfg = TranslationConfig(
        encoder=encoder_cfg,
        text_model_name=str(model_cfg["decoder"]["model_name"]),
        mbart_hidden_dim=int(model_cfg["decoder"]["hidden_dim"]),
        articulator_queries=int(encoder_cfg_raw["articulator_queries"]),
        global_queries=int(encoder_cfg_raw["global_queries"]),
        hf_cache=hf_cache,
        label_smoothing=label_smoothing,
    )
    return PgatMbartTranslationModel(translation_cfg)


def warm_start_from_alignment(
    model: PgatMbartTranslationModel,
    alignment_checkpoint: Path,
) -> None:
    """Load PGAT encoder + articulator + global summary weights from alignment.

    We only take parameters whose names match — projections trained by
    alignment (video_projection / text_projection) do not exist on the
    translation model. mBART params are always freshly initialised.
    """
    payload = load_best(alignment_checkpoint, map_location="cpu")
    alignment_state = payload["trainable_state_dict"]
    target_state = model.state_dict()
    matched = 0
    for name, tensor in alignment_state.items():
        if name in target_state and target_state[name].shape == tensor.shape:
            target_state[name] = tensor
            matched += 1
    model.load_state_dict(target_state, strict=False)
    print(f"warm-started {matched}/{len(alignment_state)} params from {alignment_checkpoint}", flush=True)


def build_parameter_groups(
    model: PgatMbartTranslationModel,
    pgat_lr: float,
    encoder_lr: float,
    decoder_lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Split parameters into groups with per-module learning rates.

    Groups:
    1. PGAT encoder side (tokenizer + articulator + global_summary) at pgat_lr.
    2. Projection + mBART encoder at encoder_lr.
    3. mBART decoder + lm_head at decoder_lr.
    """
    pgat_params: list[torch.nn.Parameter] = []
    enc_params: list[torch.nn.Parameter] = []
    dec_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("tokenizer") or name.startswith("articulator") or name.startswith("global_summary"):
            pgat_params.append(param)
        elif name.startswith("projection") or name.startswith("mbart.model.encoder"):
            enc_params.append(param)
        elif name.startswith("mbart"):
            dec_params.append(param)
        else:
            enc_params.append(param)

    return [
        {"params": pgat_params, "lr": pgat_lr, "weight_decay": weight_decay},
        {"params": enc_params, "lr": encoder_lr, "weight_decay": weight_decay},
        {"params": dec_params, "lr": decoder_lr, "weight_decay": weight_decay},
    ]


def _autocast_dtype(name: str) -> torch.dtype | None:
    key = name.strip().lower()
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16"):
        return torch.float16
    return None


def make_dataloader(
    dataset: PhoenixCachedDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_alignment_batch,
        drop_last=shuffle,
    )


@torch.inference_mode()
def evaluate_loss(
    model: PgatMbartTranslationModel,
    dataloader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for batch in dataloader:
        batch = batch_to_device(batch, device)
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                loss, stats = model(batch)
        else:
            loss, stats = model(batch)
        total_loss += stats["loss"]
        total_acc += stats["token_acc"]
        n += 1
    return {"dev_loss": total_loss / max(1, n), "dev_token_acc": total_acc / max(1, n)}


def train_translation(
    *,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    translation_config: dict[str, Any],
    output_dir: Path,
    allow_full: bool,
) -> dict[str, Any]:
    training = TranslationTrainingConfig(
        epochs=int(translation_config["training"]["epochs"]),
        micro_batch=int(translation_config["training"]["micro_batch"]),
        grad_accum=int(translation_config["training"]["grad_accum"]),
        encoder_lr=float(translation_config["training"]["encoder_learning_rate"]),
        decoder_lr=float(translation_config["training"]["decoder_learning_rate"]),
        pgat_lr=float(translation_config["training"]["pgat_learning_rate"]),
        weight_decay=float(translation_config["training"]["weight_decay"]),
        warmup_ratio=float(translation_config["training"]["warmup_ratio"]),
        label_smoothing=float(translation_config["training"]["label_smoothing"]),
        max_target_tokens=int(translation_config["training"]["max_target_tokens"]),
        mixed_precision=str(translation_config["training"]["mixed_precision"]),
        gradient_checkpointing=bool(translation_config["training"]["gradient_checkpointing"]),
        early_stop_patience=int(translation_config["early_stopping"]["patience"]),
        seed=int(translation_config.get("seed", 42)),
    )
    if not allow_full:
        raise RuntimeError("translation training requires --allow-full to run")
    set_seed(training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = _autocast_dtype(training.mixed_precision) if device.type == "cuda" else None

    from os.path import expanduser, expandvars

    def resolve(value: str) -> Path:
        return Path(expanduser(expandvars(str(value)))).resolve()

    feature_root = resolve(data_config["paths"]["feature_root"])
    manifest_path = resolve(data_config["paths"]["manifest"])
    hf_cache = resolve(data_config["paths"]["hf_cache"])
    alignment_checkpoint = resolve(translation_config["warm_start"]["alignment_checkpoint"])

    plans_root = feature_root / "plans"
    spatial_root = feature_root / "spatial"
    motion_root = feature_root / "motion"
    text_root = feature_root / "text"

    train_ds = PhoenixCachedDataset(
        plans_root, spatial_root, motion_root, text_root, manifest_path, "train"
    )
    dev_ds = PhoenixCachedDataset(
        plans_root, spatial_root, motion_root, text_root, manifest_path, "dev"
    )
    train_loader = make_dataloader(
        train_ds, batch_size=training.micro_batch, shuffle=True, num_workers=4
    )
    dev_loader = make_dataloader(
        dev_ds, batch_size=training.micro_batch, shuffle=False, num_workers=2
    )

    model = build_model(model_config, hf_cache=hf_cache, label_smoothing=training.label_smoothing)
    warm_start_from_alignment(model, alignment_checkpoint)
    if training.gradient_checkpointing:
        model.mbart.gradient_checkpointing_enable()
        # Turn off use_cache to be compatible with gradient checkpointing.
        model.mbart.config.use_cache = False
    model = model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_groups = build_parameter_groups(
        model,
        pgat_lr=training.pgat_lr,
        encoder_lr=training.encoder_lr,
        decoder_lr=training.decoder_lr,
        weight_decay=training.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    steps_per_epoch = max(1, len(train_loader) // training.grad_accum)
    total_steps = steps_per_epoch * training.epochs
    scheduler = build_scheduler(optimizer, total_steps, training.warmup_ratio)

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    best_path = output_dir / str(translation_config["checkpoint"]["best_filename"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"trainable params: {sum(p.numel() for p in trainable_params):,}",
        flush=True,
    )
    print(
        f"train={len(train_ds)} dev={len(dev_ds)} "
        f"micro_batch={training.micro_batch} grad_accum={training.grad_accum}",
        flush=True,
    )

    for epoch in range(1, training.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_micro = 0
        optimizer.zero_grad(set_to_none=True)
        start = time.monotonic()

        for step, batch in enumerate(train_loader):
            batch = batch_to_device(batch, device)
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    loss, stats = model(batch)
            else:
                loss, stats = model(batch)

            (loss / training.grad_accum).backward()
            total_loss += stats["loss"]
            total_acc += stats["token_acc"]
            n_micro += 1

            if (step + 1) % training.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        # Flush remainder if drop_last happens to leave leftover.
        if (step + 1) % training.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        elapsed = time.monotonic() - start
        avg_loss = total_loss / max(1, n_micro)
        avg_acc = total_acc / max(1, n_micro)
        peak_gib = (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        )
        print(
            f"[epoch {epoch:02d}] train loss={avg_loss:.4f} tok_acc={avg_acc:.4f} "
            f"peak_gib={peak_gib:.2f} elapsed={elapsed:.1f}s",
            flush=True,
        )

        dev_metrics = evaluate_loss(model, dev_loader, device, autocast_dtype)
        print(
            f"[epoch {epoch:02d}]  dev loss={dev_metrics['dev_loss']:.4f} "
            f"tok_acc={dev_metrics['dev_token_acc']:.4f}",
            flush=True,
        )

        record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_token_acc": avg_acc,
            **dev_metrics,
        }
        history.append(record)

        if dev_metrics["dev_loss"] < best_loss:
            best_loss = dev_metrics["dev_loss"]
            best_epoch = epoch
            stale = 0
            payload = {
                "epoch": epoch,
                "trainable_state_dict": trainable_state_dict(model),
                "model_config": model_config,
                "translation_config": translation_config,
                "dev_metrics": dev_metrics,
            }
            save_best(best_path, payload)
            print(
                f"[epoch {epoch:02d}]  saved best -> {best_path} (dev_loss={best_loss:.4f})",
                flush=True,
            )
        else:
            stale += 1
            print(
                f"[epoch {epoch:02d}]  no improvement ({stale}/{training.early_stop_patience})",
                flush=True,
            )
            if stale >= training.early_stop_patience:
                print(f"early stop after epoch {epoch} (best dev_loss={best_loss:.4f} at {best_epoch})", flush=True)
                break

    history_path = output_dir / "translation_history.json"
    history_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "best_epoch": best_epoch,
        "best_dev_loss": best_loss,
        "best_checkpoint": str(best_path),
        "history": str(history_path),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary
