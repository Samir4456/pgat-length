"""Stage 1 alignment loop using MPNet as the (frozen, cached) text encoder.

Text side reads a per-sample 768-D mpnet vector directly from the batch
(no online transformer). Same InfoNCE + hard-negative loss, same
retrieval eval, same V->T R@1 early-stopping metric as alignment_loop.py.

Consumed by scripts/17_train_alignment_mpnet.py.
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

from pgat_length.data.dataset_mpnet import MpnetPhoenixDataset, collate_mpnet_batch
from pgat_length.evaluation.retrieval import (
    RetrievalMetrics,
    video_to_text_retrieval,
)
from pgat_length.models.alignment import (
    InfoNceWithHardNegatives,
    MpnetAlignmentConfig,
    MpnetAlignmentModel,
)
from pgat_length.models.tokenizer import EncoderConfig
from pgat_length.training.checkpoint import save_best, trainable_state_dict


@dataclass(frozen=True)
class MpnetAlignmentTrainingConfig:
    epochs: int
    micro_batch: int
    grad_accum: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    temperature: float
    hard_weight: float
    early_stop_patience: int
    seed: int
    mixed_precision: str
    gradient_checkpointing: bool


def _autocast_dtype(name: str) -> torch.dtype | None:
    key = name.strip().lower()
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16"):
        return torch.float16
    return None


def _set_seed(seed: int) -> None:
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


def _build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _build_model(model_cfg: dict[str, Any], text_embedding_dim: int, alignment_dim: int) -> MpnetAlignmentModel:
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
    return MpnetAlignmentModel(MpnetAlignmentConfig(
        encoder=encoder_cfg,
        text_embedding_dim=text_embedding_dim,
        alignment_dim=alignment_dim,
        articulator_queries=int(encoder_cfg_raw["articulator_queries"]),
        global_queries=int(encoder_cfg_raw["global_queries"]),
    ))


def _make_loader(dataset: MpnetPhoenixDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_mpnet_batch,
        drop_last=shuffle,
    )


def _evaluate(
    model: MpnetAlignmentModel,
    dataloader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> RetrievalMetrics:
    model.eval()
    video_embeds: list[torch.Tensor] = []
    text_embeds: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in dataloader:
            batch = _batch_to_device(batch, device)
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    v = model.encode_video(batch)
                    t = model.encode_text(batch["mpnet_embedding"])
            else:
                v = model.encode_video(batch)
                t = model.encode_text(batch["mpnet_embedding"])
            video_embeds.append(v.float().cpu())
            text_embeds.append(t.float().cpu())
    return video_to_text_retrieval(torch.cat(video_embeds), torch.cat(text_embeds))


def train_alignment_mpnet(
    *,
    data_config: dict[str, Any],
    features_config: dict[str, Any],
    model_config: dict[str, Any],
    alignment_config: dict[str, Any],
    output_dir: Path,
    allow_full: bool,
) -> dict[str, Any]:
    training = MpnetAlignmentTrainingConfig(
        epochs=int(alignment_config["training"]["epochs"]),
        micro_batch=int(alignment_config["training"]["micro_batch"]),
        grad_accum=int(alignment_config["training"]["grad_accum"]),
        learning_rate=float(alignment_config["training"]["learning_rate"]),
        weight_decay=float(alignment_config["training"]["weight_decay"]),
        warmup_ratio=float(alignment_config["training"]["warmup_ratio"]),
        temperature=float(alignment_config["alignment"]["temperature"]),
        hard_weight=float(alignment_config["alignment"]["hard_negative_weight"]),
        early_stop_patience=int(alignment_config["early_stopping"]["patience"]),
        seed=int(alignment_config.get("seed", 42)),
        mixed_precision=str(alignment_config["training"]["mixed_precision"]),
        gradient_checkpointing=bool(alignment_config["training"]["gradient_checkpointing"]),
    )
    if not allow_full:
        raise RuntimeError("mpnet alignment training requires --allow-full to run")
    _set_seed(training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = _autocast_dtype(training.mixed_precision) if device.type == "cuda" else None

    from os.path import expanduser, expandvars

    def resolve(value: str) -> Path:
        return Path(expanduser(expandvars(str(value)))).resolve()

    feature_root = resolve(data_config["paths"]["feature_root"])
    manifest_path = resolve(data_config["paths"]["manifest"])
    plans_root = feature_root / "plans"
    spatial_root = feature_root / "spatial"
    motion_root = feature_root / "motion"
    mpnet_text_root = feature_root / "text_mpnet"

    train_ds = MpnetPhoenixDataset(plans_root, spatial_root, motion_root, mpnet_text_root, manifest_path, "train")
    dev_ds = MpnetPhoenixDataset(plans_root, spatial_root, motion_root, mpnet_text_root, manifest_path, "dev")
    train_loader = _make_loader(train_ds, batch_size=training.micro_batch, shuffle=True, num_workers=4)
    dev_loader = _make_loader(dev_ds, batch_size=training.micro_batch, shuffle=False, num_workers=2)

    text_embedding_dim = int(features_config.get("text_mpnet", {}).get("embedding_dim", 768))
    alignment_dim = int(alignment_config["alignment"]["video_embedding_dim"])
    model = _build_model(model_config, text_embedding_dim=text_embedding_dim, alignment_dim=alignment_dim).to(device)
    loss_fn = InfoNceWithHardNegatives(
        temperature=training.temperature,
        hard_weight=training.hard_weight,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader) // training.grad_accum)
    total_steps = steps_per_epoch * training.epochs
    scheduler = _build_scheduler(optimizer, total_steps, training.warmup_ratio)

    history: list[dict[str, Any]] = []
    best_r_at_1 = -1.0
    best_epoch = -1
    stale = 0
    best_path = output_dir / str(alignment_config["checkpoint"]["best_filename"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"trainable params: {sum(p.numel() for p in trainable_params):,}", flush=True)
    print(
        f"train={len(train_ds)} dev={len(dev_ds)} "
        f"micro_batch={training.micro_batch} grad_accum={training.grad_accum}",
        flush=True,
    )

    for epoch in range(1, training.epochs + 1):
        model.train()

        total_loss = 0.0
        total_stats = {"loss_infonce": 0.0, "loss_hard": 0.0, "acc_v2t": 0.0, "acc_t2v": 0.0}
        n_micro = 0
        optimizer.zero_grad(set_to_none=True)
        start = time.monotonic()

        for step, batch in enumerate(train_loader):
            batch = _batch_to_device(batch, device)
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    video = model.encode_video(batch)
                    text = model.encode_text(batch["mpnet_embedding"])
                    loss, stats = loss_fn(video, text)
            else:
                video = model.encode_video(batch)
                text = model.encode_text(batch["mpnet_embedding"])
                loss, stats = loss_fn(video, text)

            (loss / training.grad_accum).backward()
            total_loss += float(loss.detach().item())
            for key in total_stats:
                total_stats[key] += stats[key]
            n_micro += 1

            if (step + 1) % training.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if (step + 1) % training.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        elapsed = time.monotonic() - start
        avg_loss = total_loss / max(1, n_micro)
        avg_stats = {k: v / max(1, n_micro) for k, v in total_stats.items()}
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        print(
            f"[epoch {epoch:02d}] train loss={avg_loss:.4f} "
            f"infonce={avg_stats['loss_infonce']:.4f} hard={avg_stats['loss_hard']:.4f} "
            f"acc_v2t={avg_stats['acc_v2t']:.4f} acc_t2v={avg_stats['acc_t2v']:.4f} "
            f"peak_gib={peak_gib:.2f} elapsed={elapsed:.1f}s",
            flush=True,
        )

        dev_metrics = _evaluate(model, dev_loader, device, autocast_dtype)
        print(
            f"[epoch {epoch:02d}]  dev "
            f"R@1={dev_metrics.r_at_1:.4f} R@5={dev_metrics.r_at_5:.4f} "
            f"R@10={dev_metrics.r_at_10:.4f} med={dev_metrics.median_rank:.1f} "
            f"mean={dev_metrics.mean_rank:.2f}",
            flush=True,
        )

        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_stats": avg_stats,
            "dev_metrics": dev_metrics.to_dict(),
        }
        history.append(record)

        if dev_metrics.r_at_1 > best_r_at_1:
            best_r_at_1 = dev_metrics.r_at_1
            best_epoch = epoch
            stale = 0
            payload = {
                "epoch": epoch,
                "trainable_state_dict": trainable_state_dict(model),
                "model_config": model_config,
                "alignment_config": alignment_config,
                "dev_metrics": dev_metrics.to_dict(),
            }
            save_best(best_path, payload)
            print(f"[epoch {epoch:02d}]  saved best -> {best_path} (R@1={best_r_at_1:.4f})", flush=True)
        else:
            stale += 1
            print(f"[epoch {epoch:02d}]  no improvement ({stale}/{training.early_stop_patience})", flush=True)
            if stale >= training.early_stop_patience:
                print(f"early stop after epoch {epoch} (best R@1={best_r_at_1:.4f} at {best_epoch})", flush=True)
                break

    history_path = output_dir / "alignment_mpnet_history.json"
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "best_epoch": best_epoch,
        "best_v2t_recall_at_1": best_r_at_1,
        "best_checkpoint": str(best_path),
        "history": str(history_path),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary
