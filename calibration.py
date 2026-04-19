"""
ANN-to-SNN conversion utilities with conservative relative calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from spikingjelly.activation_based import ann2snn, functional, neuron

from config import CFG


MAX_QUANTILE_ELEMENTS = 8_000_000


@dataclass
class ActivationStat:
    name: str
    shape_no_batch: Tuple[int, ...]
    q_standard_all: float
    q_target_nonzero: float
    nonzero_rate: float
    overall_rate_above_target: float
    conservative_scale: float


class ActivationCollector:
    def __init__(self, model: nn.Module) -> None:
        self.hooks = []
        self.storage: Dict[str, List[torch.Tensor]] = {}
        self.relu_names: List[Tuple[str, nn.Module]] = []
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                self.relu_names.append((name, module))
                self.storage[name] = []

    def register(self) -> None:
        for name, module in self.relu_names:
            self.hooks.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_, __, output):
            self.storage[name].append(output.detach().cpu().float())
        return hook

    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get(self) -> Dict[str, torch.Tensor]:
        return {
            name: torch.cat(tensors, dim=0)
            for name, tensors in self.storage.items()
            if tensors
        }


def _subsample_if_needed(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= MAX_QUANTILE_ELEMENTS:
        return x
    gen = torch.Generator().manual_seed(42)
    idx = torch.randperm(x.numel(), generator=gen)[:MAX_QUANTILE_ELEMENTS]
    return x[idx]


def _ordered_snn_neurons(snn_model):
    return [
        (name, module)
        for name, module in snn_model.named_modules()
        if isinstance(module, (neuron.IFNode, neuron.LIFNode))
    ]


def _collect_ann_stats(ann_model, calib_loader, device, calib_batches: int, target_firing_rate: float):
    collector = ActivationCollector(ann_model.to(device))
    collector.register()
    first_images = None
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(calib_loader):
            if first_images is None:
                first_images = images[:2].to(device)
            ann_model(images.to(device))
            if batch_idx + 1 >= calib_batches:
                break
    collector.remove()

    acts = collector.get()
    stats: List[ActivationStat] = []
    target_quantile = 1.0 - target_firing_rate

    for name, tensor in acts.items():
        flat = tensor.flatten()
        flat = _subsample_if_needed(flat)
        nonzero = flat[flat > 0]

        q_standard_all = float(torch.quantile(flat, CFG.snn.STANDARD_PERCENTILE / 100.0))
        if nonzero.numel() == 0:
            q_target_nonzero = q_standard_all
            nonzero_rate = 0.0
        else:
            nonzero = _subsample_if_needed(nonzero)
            q_target_nonzero = float(torch.quantile(nonzero, target_quantile))
            nonzero_rate = float((flat > 0).float().mean())

        ratio = q_target_nonzero / max(q_standard_all, 1e-6)
        conservative_scale = float(min(max(ratio, CFG.snn.MIN_SCALE), CFG.snn.MAX_SCALE))
        overall_rate_above_target = float((flat > q_target_nonzero).float().mean())
        stats.append(
            ActivationStat(
                name=name,
                shape_no_batch=tuple(tensor.shape[1:]),
                q_standard_all=q_standard_all,
                q_target_nonzero=q_target_nonzero,
                nonzero_rate=nonzero_rate,
                overall_rate_above_target=overall_rate_above_target,
                conservative_scale=conservative_scale,
            )
        )

    return stats, first_images


def _collect_snn_shapes(snn_model, sample_images: torch.Tensor, device):
    hooks = []
    shapes = {}

    def make_hook(name):
        def hook(_, __, output):
            shapes[name] = tuple(output.shape[1:])
        return hook

    for name, module in _ordered_snn_neurons(snn_model):
        hooks.append(module.register_forward_hook(make_hook(name)))

    functional.reset_net(snn_model)
    with torch.no_grad():
        _ = snn_model(sample_images.to(device))
    functional.reset_net(snn_model)

    for hook in hooks:
        hook.remove()

    return shapes


def convert_standard(ann_model, calib_loader, device, percentile: float | None = None):
    percentile = CFG.snn.STANDARD_PERCENTILE if percentile is None else percentile
    ann_model.eval()
    converter = ann2snn.Converter(
        mode=f"{percentile}%",
        dataloader=calib_loader,
        device=device,
    )
    snn_model = converter(ann_model).to(device)
    print(f"standard conversion complete at {percentile}%")
    return snn_model


def convert_with_calibration(
    ann_model,
    calib_loader,
    device,
    target_firing_rate: float | None = None,
    calib_batches: int | None = None,
    verbose: bool = True,
):
    target_firing_rate = CFG.snn.TARGET_FIRING_RATE if target_firing_rate is None else target_firing_rate
    calib_batches = CFG.snn.CALIB_BATCHES if calib_batches is None else calib_batches

    ann_model.eval()
    snn_model = convert_standard(ann_model, calib_loader, device, CFG.snn.STANDARD_PERCENTILE)
    ann_stats, sample_images = _collect_ann_stats(ann_model, calib_loader, device, calib_batches, target_firing_rate)
    snn_neurons = _ordered_snn_neurons(snn_model)
    snn_shapes = _collect_snn_shapes(snn_model, sample_images, device)

    if len(ann_stats) != len(snn_neurons):
        raise RuntimeError(
            f"ANN ReLU count ({len(ann_stats)}) does not match SNN IF count ({len(snn_neurons)})."
        )

    calibration_report = []
    if verbose:
        print(f"\nconservative relative calibration target = {target_firing_rate:.2f}")
        print(f'{"ANN ReLU":<28} {"SNN IF":<24} {"Scale":>7} {"Overall>q":>10}')

    for stat, (if_name, if_node) in zip(ann_stats, snn_neurons):
        snn_shape = snn_shapes.get(if_name)
        if snn_shape != stat.shape_no_batch:
            raise RuntimeError(
                f"Shape mismatch in calibration mapping: {stat.name} {stat.shape_no_batch} vs {if_name} {snn_shape}"
            )
        base_threshold = float(getattr(if_node, "v_threshold", 1.0))
        new_threshold = base_threshold * stat.conservative_scale
        if_node.v_threshold = new_threshold
        row = {
            "ann_relu": stat.name,
            "snn_if": if_name,
            "base_threshold": base_threshold,
            "new_threshold": new_threshold,
            "scale": stat.conservative_scale,
            "overall_rate_above_target": stat.overall_rate_above_target,
            "nonzero_rate": stat.nonzero_rate,
        }
        calibration_report.append(row)
        if verbose:
            print(
                f"{stat.name:<28} {if_name:<24} "
                f"{stat.conservative_scale:>7.3f} {stat.overall_rate_above_target*100:>9.2f}%"
            )

    return snn_model, calibration_report
