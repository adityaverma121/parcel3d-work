"""
Energy and compute accounting utilities.

SNN SynOps are approximated as:
  sum_over_layers(next_dense_MACs_per_image * spike_rate_of_previous_IF_layer)

This is still an approximation, but it is substantially more defensible than
pricing raw emitted spike counts directly as synaptic operations.
"""

from __future__ import annotations

import json
from typing import Dict, List

import torch
import torch.nn as nn

from config import CFG


def compute_cnn_energy(model, device, img_size: int | None = None):
    img_size = CFG.data.IMG_SIZE if img_size is None else img_size
    from thop import profile as thop_profile

    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        macs, params = thop_profile(model, inputs=(dummy,), verbose=False)
    energy_pj = macs * CFG.energy.E_MAC_PJ
    return {
        "macs": float(macs),
        "params": float(params),
        "energy_pJ": float(energy_pj),
        "energy_uJ": float(energy_pj / 1e6),
        "op_type": "MAC",
    }


def build_activation_cost_model(model, device, img_size: int | None = None):
    """
    Returns the dense MAC cost of the compute layer following each ReLU.
    """
    img_size = CFG.data.IMG_SIZE if img_size is None else img_size
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    ordered_relus: List[str] = []
    named_modules = list(model.named_modules())
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            ordered_relus.append(name)

    per_compute_macs: Dict[str, float] = {}

    def add_hooks():
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                def make_hook(layer_name):
                    def hook(m, inputs, output):
                        x = inputs[0]
                        if isinstance(m, nn.Conv2d):
                            batch_size = x.shape[0]
                            out = output
                            out_h, out_w = out.shape[-2], out.shape[-1]
                            kernel_h, kernel_w = m.kernel_size
                            in_per_group = m.in_channels // m.groups
                            macs = (
                                batch_size
                                * m.out_channels
                                * out_h
                                * out_w
                                * in_per_group
                                * kernel_h
                                * kernel_w
                            )
                        else:
                            batch_size = x.shape[0]
                            macs = batch_size * m.in_features * m.out_features
                        per_compute_macs[layer_name] = float(macs / batch_size)
                    return hook
                hooks.append(module.register_forward_hook(make_hook(name)))
        return hooks

    hooks = add_hooks()
    with torch.no_grad():
        _ = model(dummy)
    for hook in hooks:
        hook.remove()

    next_compute_names = []
    next_compute_macs = []
    for relu_name in ordered_relus:
        relu_index = next(i for i, (name, _) in enumerate(named_modules) if name == relu_name)
        chosen_name = None
        for name, module in named_modules[relu_index + 1:]:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                chosen_name = name
                break
        if chosen_name is None:
            raise RuntimeError(f"Could not find downstream compute layer for activation {relu_name}")
        next_compute_names.append(chosen_name)
        next_compute_macs.append(per_compute_macs[chosen_name])

    return {
        "activation_names": ordered_relus,
        "next_compute_names": next_compute_names,
        "next_compute_macs": next_compute_macs,
    }


def compute_snn_energy_from_results(snn_results: Dict[int, Dict], cost_model: Dict):
    energies = {}
    dense_sum = sum(cost_model["next_compute_macs"])
    for T, result in snn_results.items():
        layer_rates = result.get("spike_rates", {})
        synops = 0.0
        per_layer_synops = {}
        for act_name, next_name, next_macs, snn_name in zip(
            cost_model["activation_names"],
            cost_model["next_compute_names"],
            cost_model["next_compute_macs"],
            result["if_node_names"],
        ):
            rate = layer_rates.get(snn_name, 0.0)
            layer_synops = float(next_macs * rate)
            per_layer_synops[snn_name] = layer_synops
            synops += layer_synops
        energy_pj = synops * CFG.energy.E_AC_PJ
        energies[T] = {
            "T": T,
            "dense_macs_reference": float(dense_sum),
            "synops": float(synops),
            "per_layer_synops": per_layer_synops,
            "energy_pJ": float(energy_pj),
            "energy_uJ": float(energy_pj / 1e6),
            "sparsity": float(1.0 - result["mean_spike_rate"]),
            "spike_rate": float(result["mean_spike_rate"]),
            "per_layer_rates": layer_rates,
            "op_type": "AC",
        }
    return energies


def dump_results_summary(path: str, payload: Dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def print_comparison_table(cnn_energy, cnn_metrics, results_std, results_ours, energies_std, energies_ours):
    cnn_uJ = cnn_energy["energy_uJ"]
    print("\n" + "=" * 92)
    print("TABLE: CNN vs SNN")
    print("=" * 92)
    print(f'{"Model":<26} {"T":>5} {"Acc":>8} {"F1":>8} {"AUC":>8} {"Energy uJ":>12} {"Ratio":>8}')
    print(
        f'{"CNN":<26} {"-":>5} '
        f'{cnn_metrics["accuracy"]*100:7.2f}% '
        f'{cnn_metrics["f1_macro"]*100:7.2f}% '
        f'{cnn_metrics["auc_roc"]*100:7.2f}% '
        f'{cnn_uJ:11.4f} {"1.0x":>8}'
    )
    for label, results, energies in [
        ("Standard", results_std, energies_std),
        ("Calibrated", results_ours, energies_ours),
    ]:
        for T in sorted(results):
            energy = energies[T]["energy_uJ"]
            ratio = cnn_uJ / energy if energy > 0 else 0.0
            print(
                f"{label:<26} {T:5d} "
                f'{results[T]["accuracy"]*100:7.2f}% '
                f'{results[T]["f1_macro"]*100:7.2f}% '
                f'{results[T]["auc_roc"]*100:7.2f}% '
                f"{energy:11.4f} {ratio:7.1f}x"
            )
