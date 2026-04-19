"""
Publication-friendly plotting utilities.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import CFG


sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
FIG_DIR = Path(CFG.FIG_DIR)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str):
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(log_csv: str):
    rows = []
    with open(log_csv, "r") as f:
        for row in csv.DictReader(f):
            row["seed"] = int(row["seed"])
            row["stage"] = int(row["stage"])
            row["epoch"] = int(row["epoch"])
            for key in ["tr_loss", "tr_acc", "tr_f1", "va_loss", "va_acc", "va_f1", "seconds"]:
                row[key] = float(row[key])
            rows.append(row)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for seed in sorted({r["seed"] for r in rows}):
        seed_rows = [r for r in rows if r["seed"] == seed]
        x = np.arange(1, len(seed_rows) + 1)
        axes[0].plot(x, [r["va_loss"] for r in seed_rows], label=f"seed {seed}")
        axes[1].plot(x, [r["va_f1"] for r in seed_rows], label=f"seed {seed}")

    axes[0].set_title("Validation loss")
    axes[1].set_title("Validation macro F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "fig1_training_curves")


def plot_confusion_matrices(cnn_cm, snn_cm, best_T):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, cm, title in [
        (axes[0], cnn_cm, "CNN"),
        (axes[1], snn_cm, f"SNN calibrated (T={best_T})"),
    ]:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    _save(fig, "fig2_confusion_matrices")


def plot_accuracy_vs_T(results_std, results_ours, cnn_accuracy, cnn_f1):
    T_values = sorted(results_std.keys())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    metrics = [("accuracy", "Accuracy (%)", cnn_accuracy * 100), ("f1_macro", "Macro F1 (%)", cnn_f1 * 100)]
    for ax, (metric, ylabel, ref) in zip(axes, metrics):
        ax.plot(T_values, [results_std[T][metric] * 100 for T in T_values], "s--", label="standard")
        ax.plot(T_values, [results_ours[T][metric] * 100 for T in T_values], "o-", label="calibrated")
        ax.axhline(ref, linestyle=":", color="black", label="cnn")
        ax.set_xscale("log", base=2)
        ax.set_xticks(T_values)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Time steps T")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "fig3_accuracy_vs_T")


def plot_pareto_curve(cnn_energy, cnn_metrics, energies_std, results_std, energies_ours, results_ours):
    T_values = sorted(results_std.keys())
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot(
        [energies_std[T]["energy_uJ"] for T in T_values],
        [results_std[T]["accuracy"] * 100 for T in T_values],
        "s--",
        label="standard",
    )
    ax.plot(
        [energies_ours[T]["energy_uJ"] for T in T_values],
        [results_ours[T]["accuracy"] * 100 for T in T_values],
        "o-",
        label="calibrated",
    )
    ax.scatter([cnn_energy["energy_uJ"]], [cnn_metrics["accuracy"] * 100], marker="*", s=170, label="cnn")
    ax.set_xlabel("Energy per image (uJ)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "fig4_pareto_curve")


def plot_spike_rate_heatmap(energies_ours, T_values):
    T_values = sorted(T_values)
    layer_names = list(energies_ours[T_values[0]].get("per_layer_rates", {}).keys())
    if not layer_names:
        print("skip spike heatmap: no layer rates")
        return
    data = np.array(
        [[energies_ours[T]["per_layer_rates"].get(name, 0.0) * 100 for name in layer_names] for T in T_values]
    )
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    sns.heatmap(data, cmap="YlOrRd", ax=ax, xticklabels=layer_names, yticklabels=[f"T={T}" for T in T_values])
    ax.set_title("Per-layer spike rate (%)")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout()
    _save(fig, "fig5_spike_rate_heatmap")


def plot_ablation_table(results_std, results_ours, energies_std, energies_ours, cnn_energy):
    T_values = sorted(results_std.keys())
    x = np.arange(len(T_values))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    axes[0].bar(x - width / 2, [results_std[T]["f1_macro"] * 100 for T in T_values], width, label="standard")
    axes[0].bar(x + width / 2, [results_ours[T]["f1_macro"] * 100 for T in T_values], width, label="calibrated")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"T={T}" for T in T_values])
    axes[0].set_ylabel("Macro F1 (%)")
    axes[0].legend(fontsize=7)

    axes[1].bar(
        x - width / 2,
        [cnn_energy["energy_uJ"] / max(energies_std[T]["energy_uJ"], 1e-9) for T in T_values],
        width,
        label="standard",
    )
    axes[1].bar(
        x + width / 2,
        [cnn_energy["energy_uJ"] / max(energies_ours[T]["energy_uJ"], 1e-9) for T in T_values],
        width,
        label="calibrated",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"T={T}" for T in T_values])
    axes[1].set_ylabel("Energy reduction (x)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "fig6_ablation")
