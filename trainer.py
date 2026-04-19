"""
Training utilities for reproducible multi-seed Parcel3D experiments.
"""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import CFG


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def step(self, score: float) -> bool:
        if self.best is None or score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool, use_amp: bool):
    model.train(train)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    context = torch.enable_grad() if train else torch.no_grad()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    n_batches = 0

    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(amp_device, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, f1


def _append_log_row(rows: List[Dict], **kwargs) -> None:
    rows.append(kwargs)


def _save_log_csv(rows: List[Dict]) -> None:
    if not rows:
        return
    Path(CFG.cnn.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    with open(CFG.cnn.LOG_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_seed_summary(seed_results: Dict[int, float], best_checkpoint: str) -> None:
    payload = {
        "seed_results": seed_results,
        "mean_val_f1": float(np.mean(list(seed_results.values()))),
        "std_val_f1": float(np.std(list(seed_results.values()))),
        "best_checkpoint": best_checkpoint,
    }
    with open(Path(CFG.cnn.CHECKPOINT_DIR) / "seed_results_summary.json", "w") as f:
        json.dump(payload, f, indent=2)


def train_single_seed(model, loaders, device, seed: int):
    seed_everything(seed)
    ckpt_path = Path(CFG.cnn.CHECKPOINT_DIR) / f"best_seed_{seed}.pth"

    use_amp = CFG.cnn.USE_AMP and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.cnn.LABEL_SMOOTHING).to(device)
    logs: List[Dict] = []
    best_f1 = -1.0

    print(f"\n[seed {seed}] stage 1")
    model.freeze_stage1()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.cnn.STAGE1_LR,
        weight_decay=CFG.cnn.STAGE1_WD,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.cnn.STAGE1_EPOCHS, eta_min=CFG.cnn.ETA_MIN)

    for epoch in range(1, CFG.cnn.STAGE1_EPOCHS + 1):
        start = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, loaders["train"], criterion, optimizer, scaler, device, True, use_amp)
        va_loss, va_acc, va_f1 = run_epoch(model, loaders["val"], criterion, optimizer, scaler, device, False, use_amp)
        scheduler.step()
        secs = time.time() - start
        print(f"  s1 ep {epoch:02d}/{CFG.cnn.STAGE1_EPOCHS} | va_f1={va_f1:.4f} | {secs:.0f}s")
        _append_log_row(
            logs,
            seed=seed,
            stage=1,
            epoch=epoch,
            tr_loss=tr_loss,
            tr_acc=tr_acc,
            tr_f1=tr_f1,
            va_loss=va_loss,
            va_acc=va_acc,
            va_f1=va_f1,
            seconds=secs,
        )
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), ckpt_path)

    print(f"\n[seed {seed}] stage 2")
    model.unfreeze_stage2()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.cnn.STAGE2_LR,
        weight_decay=CFG.cnn.STAGE2_WD,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.cnn.STAGE2_EPOCHS, eta_min=CFG.cnn.ETA_MIN)
    stopper = EarlyStopping(CFG.cnn.PATIENCE)

    for epoch in range(1, CFG.cnn.STAGE2_EPOCHS + 1):
        start = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, loaders["train"], criterion, optimizer, scaler, device, True, use_amp)
        va_loss, va_acc, va_f1 = run_epoch(model, loaders["val"], criterion, optimizer, scaler, device, False, use_amp)
        scheduler.step()
        secs = time.time() - start
        print(f"  s2 ep {epoch:02d}/{CFG.cnn.STAGE2_EPOCHS} | va_f1={va_f1:.4f} | {secs:.0f}s")
        _append_log_row(
            logs,
            seed=seed,
            stage=2,
            epoch=epoch,
            tr_loss=tr_loss,
            tr_acc=tr_acc,
            tr_f1=tr_f1,
            va_loss=va_loss,
            va_acc=va_acc,
            va_f1=va_f1,
            seconds=secs,
        )
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), ckpt_path)
        if stopper.step(va_f1):
            print(f"  early stop at epoch {epoch}")
            break

    return best_f1, str(ckpt_path), logs


def train_multi_seed(model_class, loaders, device):
    Path(CFG.cnn.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    all_logs: List[Dict] = []
    seed_results: Dict[int, float] = {}
    best_overall = -1.0
    best_checkpoint = None

    for seed in CFG.cnn.SEEDS:
        model = model_class(
            num_classes=CFG.NUM_CLASSES,
            pretrained=True,
            dropout=CFG.cnn.DROPOUT,
        ).to(device)
        model.verify_ann2snn_compatibility()
        best_f1, ckpt_path, logs = train_single_seed(model, loaders, device, seed)
        all_logs.extend(logs)
        seed_results[seed] = best_f1
        if best_f1 > best_overall:
            best_overall = best_f1
            best_checkpoint = ckpt_path
        print(f"[seed {seed}] best val f1={best_f1:.4f}")

    _save_log_csv(all_logs)
    _save_seed_summary(seed_results, best_checkpoint)
    return best_checkpoint, all_logs, seed_results
