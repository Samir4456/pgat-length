"""Stage 2 Qwen translation training loop.

Video encoder + variable-length prefix (identical to mBART path) is
warm-started from an alignment checkpoint. Decoder is Qwen2.5-3B with
NF4 quantization and LoRA adapters.

Parameter groups:
    - PGAT (tokenizer + articulator + global_summary)   pgat_lr
    - projection                                        projection_lr
    - LoRA adapters                                     lora_lr

Loss: causal-LM next-token CE on target tokens only (prompt + prefix
positions are -100). Label smoothing is applied by the loss config on
the Qwen loss call (label_smoothing kwarg on Qwen's forward is not
supported, so we replicate label smoothing manually only if requested).
For now we use Qwen's built-in cross entropy (no smoothing) to match
PGAT-v1's recipe -- the CLAUDE.md-recorded PGAT-v1 result used the same.
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

from pgat_length.data.dataset_qwen import QwenPhoenixDataset, collate_qwen_batch
from pgat_length.models.qwen_translation import (
    PgatQwenTranslationModel,
    QwenTranslationConfig,
)
from pgat_length.models.tokenizer import EncoderConfig
from pgat_length.training.checkpoint import load_best, save_best, trainable_state_dict
from pgat_length.training.qwen_runtime import (
    cosine_lr_lambda,
    get_lora_state_dict,
    load_qwen_qlora_decoder,
)


@dataclass(frozen=True)
class QwenTrainingConfig:
    epochs: int
    micro_batch: int
    grad_accum: int
    pgat_lr: float
    projection_lr: float
    lora_lr: float
    weight_decay: float
    warmup_ratio: float
    mixed_precision: str
    gradient_checkpointing: bool
    early_stop_patience: int
    seed: int


def _autocast_dtype(name: str) -> torch.dtype | None:
    key = name.strip().lower()
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16"):
        return torch.float16
    return None


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def build_model(model_cfg: dict[str, Any], qlora_cfg: dict[str, Any], hf_cache: Path) -> PgatQwenTranslationModel:
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
    return PgatQwenTranslationModel(QwenTranslationConfig(
        encoder=encoder_cfg,
        qwen_hidden_dim=int(model_cfg["decoder"]["hidden_dim"]),
        articulator_queries=int(encoder_cfg_raw["articulator_queries"]),
        global_queries=int(encoder_cfg_raw["global_queries"]),
        qlora=qlora_cfg,
        hf_cache=hf_cache,
    ))


def warm_start_from_alignment(model: PgatQwenTranslationModel, alignment_checkpoint: Path) -> int:
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
    return matched


def build_parameter_groups(
    model: PgatQwenTranslationModel,
    pgat_lr: float,
    projection_lr: float,
    lora_lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    pgat_params: list[torch.nn.Parameter] = []
    proj_params: list[torch.nn.Parameter] = []
    lora_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("tokenizer") or name.startswith("articulator") or name.startswith("global_summary"):
            pgat_params.append(param)
        elif name.startswith("projection"):
            proj_params.append(param)
        elif "lora_" in name:
            lora_params.append(param)
        # Non-LoRA decoder params must all be frozen at this point (nf4 base).

    if not lora_params:
        raise RuntimeError("No LoRA trainable parameters found on Qwen decoder")

    return [
        {"params": pgat_params, "lr": pgat_lr, "weight_decay": weight_decay, "name": "pgat"},
        {"params": proj_params, "lr": projection_lr, "weight_decay": weight_decay, "name": "projection"},
        {"params": lora_params, "lr": lora_lr, "weight_decay": 0.0, "name": "lora"},
    ]


def make_dataloader(dataset: QwenPhoenixDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_qwen_batch,
        drop_last=shuffle,
    )


@torch.inference_mode()
def evaluate_loss(
    model: PgatQwenTranslationModel,
    dataloader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for batch in dataloader:
        batch = _batch_to_device(batch, device)
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                loss, stats = model(batch)
        else:
            loss, stats = model(batch)
        total_loss += stats["loss"]
        total_acc += stats["token_acc"]
        n += 1
    return {"dev_loss": total_loss / max(1, n), "dev_token_acc": total_acc / max(1, n)}


def train_qwen_translation(
    *,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    translation_config: dict[str, Any],
    output_dir: Path,
    allow_full: bool,
) -> dict[str, Any]:
    training = QwenTrainingConfig(
        epochs=int(translation_config["training"]["epochs"]),
        micro_batch=int(translation_config["training"]["micro_batch"]),
        grad_accum=int(translation_config["training"]["grad_accum"]),
        pgat_lr=float(translation_config["training"]["pgat_learning_rate"]),
        projection_lr=float(translation_config["training"]["projection_learning_rate"]),
        lora_lr=float(translation_config["training"]["lora_learning_rate"]),
        weight_decay=float(translation_config["training"]["weight_decay"]),
        warmup_ratio=float(translation_config["training"]["warmup_ratio"]),
        mixed_precision=str(translation_config["training"]["mixed_precision"]),
        gradient_checkpointing=bool(translation_config["training"]["gradient_checkpointing"]),
        early_stop_patience=int(translation_config["early_stopping"]["patience"]),
        seed=int(translation_config.get("seed", 42)),
    )
    if not allow_full:
        raise RuntimeError("Qwen translation training requires --allow-full to run")
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
    qwen_text_root = feature_root / "text_qwen"

    train_ds = QwenPhoenixDataset(plans_root, spatial_root, motion_root, qwen_text_root, manifest_path, "train")
    dev_ds = QwenPhoenixDataset(plans_root, spatial_root, motion_root, qwen_text_root, manifest_path, "dev")
    train_loader = make_dataloader(train_ds, batch_size=training.micro_batch, shuffle=True, num_workers=4)
    dev_loader = make_dataloader(dev_ds, batch_size=training.micro_batch, shuffle=False, num_workers=2)

    model = build_model(model_config, translation_config["qlora"], hf_cache=hf_cache)
    warm_start_from_alignment(model, alignment_checkpoint)

    # Move visual side to GPU first, then attach QLoRA decoder (bnb loads to
    # device_map={"": 0} directly and cannot be moved after construction).
    model = model.to(device)
    decoder, _ = load_qwen_qlora_decoder(translation_config["qlora"], hf_cache_dir=str(hf_cache))
    model.attach_qlora_decoder(decoder, decoder_dtype=torch.bfloat16)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_groups = build_parameter_groups(
        model,
        pgat_lr=training.pgat_lr,
        projection_lr=training.projection_lr,
        lora_lr=training.lora_lr,
        weight_decay=training.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    steps_per_epoch = max(1, len(train_loader) // training.grad_accum)
    total_steps = steps_per_epoch * training.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr_lambda(total_steps, training.warmup_ratio))

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    best_path = output_dir / str(translation_config["checkpoint"]["best_filename"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"trainable params: {sum(p.numel() for p in trainable_params):,}", flush=True)
    print(
        f"train={len(train_ds)} dev={len(dev_ds)} "
        f"micro_batch={training.micro_batch} grad_accum={training.grad_accum}",
        flush=True,
    )

    log_every = max(1, len(train_loader) // 20)
    global_step = 0
    for epoch in range(1, training.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_micro = 0
        optimizer.zero_grad(set_to_none=True)
        start = time.monotonic()
        window_loss = 0.0
        window_n = 0

        for step, batch in enumerate(train_loader):
            batch = _batch_to_device(batch, device)
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    loss, stats = model(batch)
            else:
                loss, stats = model(batch)

            (loss / training.grad_accum).backward()
            total_loss += stats["loss"]
            total_acc += stats["token_acc"]
            window_loss += stats["loss"]
            window_n += 1
            n_micro += 1
            global_step += 1

            if (step + 1) % training.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % log_every == 0:
                elapsed = max(time.monotonic() - start, 1e-6)
                samples_per_s = (step + 1) * training.micro_batch / elapsed
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  e{epoch:02d} step {step+1:4d}/{len(train_loader)} "
                    f"loss={(window_loss / max(1, window_n)):.4f} "
                    f"lr={lr:.2e} {samples_per_s:.1f}samp/s",
                    flush=True,
                )
                window_loss = 0.0
                window_n = 0

        if (step + 1) % training.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        elapsed = time.monotonic() - start
        avg_loss = total_loss / max(1, n_micro)
        avg_acc = total_acc / max(1, n_micro)
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
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

        record = {"epoch": epoch, "train_loss": avg_loss, "train_token_acc": avg_acc, **dev_metrics}
        history.append(record)

        if dev_metrics["dev_loss"] < best_loss:
            best_loss = dev_metrics["dev_loss"]
            best_epoch = epoch
            stale = 0
            payload = {
                "epoch": epoch,
                # PGAT/projection weights (visual side + projection): trainable_state_dict
                # skips the frozen quantized Qwen base.
                "trainable_state_dict": trainable_state_dict(model),
                "lora_state_dict": get_lora_state_dict(model.decoder),  # type: ignore[arg-type]
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

    history_path = output_dir / "translation_qwen_history.json"
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "best_epoch": best_epoch,
        "best_dev_loss": best_loss,
        "best_checkpoint": str(best_path),
        "history": str(history_path),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary
