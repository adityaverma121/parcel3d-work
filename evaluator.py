"""
CNN and SNN evaluation helpers.
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from spikingjelly.activation_based import functional, neuron

from config import CFG


@torch.no_grad()
def evaluate_cnn(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    latencies = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        start = time.perf_counter()
        logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) / images.shape[0] * 1000)

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = logits.argmax(dim=1).detach().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "auc_roc": roc_auc_score(all_labels, all_probs),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "report": classification_report(
            all_labels,
            all_preds,
            target_names=CFG.CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "latency_ms": float(np.mean(latencies[1:])) if len(latencies) > 1 else float(latencies[0]),
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


@torch.no_grad()
def evaluate_snn(snn_model, loader, device, T: int):
    snn_model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    latencies = []

    layer_spikes: Dict[str, float] = {}
    layer_elements: Dict[str, float] = {}
    hooks = []
    if_node_names: List[str] = []

    def make_hook(name):
        def hook(_, __, output):
            spikes = output.detach().float().sum().item()
            elems = float(output.numel())
            layer_spikes[name] = layer_spikes.get(name, 0.0) + spikes
            layer_elements[name] = layer_elements.get(name, 0.0) + elems
        return hook

    for name, module in snn_model.named_modules():
        if isinstance(module, (neuron.IFNode, neuron.LIFNode)):
            if_node_names.append(name)
            hooks.append(module.register_forward_hook(make_hook(name)))

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        batch_size = images.shape[0]
        functional.reset_net(snn_model)

        accum = torch.zeros(batch_size, CFG.NUM_CLASSES, device=device)
        start = time.perf_counter()
        for _ in range(T):
            accum += snn_model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) / batch_size * 1000)

        logits = accum / T
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = logits.argmax(dim=1).detach().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())

    for hook in hooks:
        hook.remove()

    per_layer_rates = {}
    for name in if_node_names:
        per_layer_rates[name] = layer_spikes.get(name, 0.0) / max(layer_elements.get(name, 1.0), 1.0)
    mean_spike_rate = float(np.mean(list(per_layer_rates.values()))) if per_layer_rates else 0.0

    return {
        "T": T,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "auc_roc": roc_auc_score(all_labels, all_probs),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "report": classification_report(
            all_labels,
            all_preds,
            target_names=CFG.CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "latency_ms": float(np.mean(latencies[1:])) if len(latencies) > 1 else float(latencies[0]),
        "spike_rates": per_layer_rates,
        "mean_spike_rate": mean_spike_rate,
        "if_node_names": if_node_names,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


def sweep_T_values(snn_model, loader, device, T_values, label: str = ""):
    results = {}
    tag = f" [{label}]" if label else ""
    print(f"\nT sweep{tag}: {T_values}")
    print(f'{"T":>5} {"Acc":>8} {"F1":>8} {"AUC":>8} {"Rate":>10} {"Latency":>10}')
    for T in T_values:
        result = evaluate_snn(snn_model, loader, device, T)
        results[T] = result
        print(
            f"{T:5d} "
            f"{result['accuracy']*100:7.2f}% "
            f"{result['f1_macro']*100:7.2f}% "
            f"{result['auc_roc']*100:7.2f}% "
            f"{result['mean_spike_rate']*100:9.2f}% "
            f"{result['latency_ms']:9.2f}"
        )
    return results
