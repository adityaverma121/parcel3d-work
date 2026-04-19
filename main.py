"""
Main experiment entrypoint for research-grade Parcel3D training and ANN-to-SNN evaluation.
"""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch

from calibration import convert_standard, convert_with_calibration
from config import CFG
from dataset_parcel3d import get_dataloaders
from energy_metrics import (
    build_activation_cost_model,
    compute_cnn_energy,
    compute_snn_energy_from_results,
    dump_results_summary,
    print_comparison_table,
)
from evaluator import evaluate_cnn, sweep_T_values
from models import ParcelVGG
from plot_results import (
    plot_ablation_table,
    plot_accuracy_vs_T,
    plot_confusion_matrices,
    plot_pareto_curve,
    plot_spike_rate_heatmap,
    plot_training_curves,
)
from trainer import train_multi_seed


SKIP_CNN_TRAINING = False
SKIP_STANDARD = False
SKIP_CALIBRATED = False
SKIP_FIGURES = False


def save_pkl(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def get_device():
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("device: Apple MPS")
        return torch.device("mps")
    print("device: CPU")
    return torch.device("cpu")


def main():
    CFG.dump_metadata()
    torch.set_float32_matmul_precision("high")
    device = get_device()

    loaders, _ = get_dataloaders(
        data_root=CFG.data.DATA_ROOT,
        batch_size=CFG.data.BATCH_SIZE,
        num_workers=CFG.data.NUM_WORKERS,
        img_size=CFG.data.IMG_SIZE,
        pin_memory=CFG.data.PIN_MEMORY and device.type == "cuda",
    )

    best_ckpt_path = Path(CFG.cnn.CHECKPOINT_DIR) / "best_cnn_overall.pth"
    seed_results_path = Path(CFG.cnn.CHECKPOINT_DIR) / "seed_results.pkl"

    if SKIP_CNN_TRAINING and best_ckpt_path.exists():
        print("skipping CNN training and loading saved seed results")
        seed_results = load_pkl(seed_results_path)
    else:
        best_ckpt, _, seed_results = train_multi_seed(ParcelVGG, loaders, device)
        shutil.copy(best_ckpt, best_ckpt_path)
        save_pkl(seed_results, seed_results_path)

    cnn = ParcelVGG(num_classes=CFG.NUM_CLASSES, pretrained=False, dropout=CFG.cnn.DROPOUT).to(device)
    cnn.unfreeze_stage2()
    cnn.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
    cnn.eval()

    print("\n[cnn] test evaluation")
    cnn_metrics = evaluate_cnn(cnn, loaders["test"], device)
    print(f"accuracy={cnn_metrics['accuracy']*100:.2f}%")
    print(f"f1_macro={cnn_metrics['f1_macro']*100:.2f}%")
    print(f"auc_roc={cnn_metrics['auc_roc']*100:.2f}%")
    print(
        f"val_f1 across seeds = {np.mean(list(seed_results.values())):.4f} ± "
        f"{np.std(list(seed_results.values())):.4f}"
    )

    save_json(
        {
            "seed_results": seed_results,
            "cnn_test": {
                "accuracy": cnn_metrics["accuracy"],
                "f1_macro": cnn_metrics["f1_macro"],
                "auc_roc": cnn_metrics["auc_roc"],
                "latency_ms": cnn_metrics["latency_ms"],
            },
        },
        CFG.cnn.METRICS_JSON,
    )

    cnn_energy = compute_cnn_energy(cnn, device, CFG.data.IMG_SIZE)
    print(f"cnn energy = {cnn_energy['energy_uJ']:.4f} uJ/img")
    cost_model = build_activation_cost_model(cnn, device, CFG.data.IMG_SIZE)

    std_results_path = Path(CFG.data.OUTPUT_DIR) / "results_standard.pkl"
    cal_results_path = Path(CFG.data.OUTPUT_DIR) / "results_calibrated.pkl"

    if SKIP_STANDARD and std_results_path.exists():
        results_std = load_pkl(std_results_path)
    else:
        snn_std = convert_standard(cnn, loaders["val"], device)
        results_std = sweep_T_values(snn_std, loaders["test"], device, CFG.snn.T_VALUES, label="standard")
        save_pkl(results_std, std_results_path)

    if SKIP_CALIBRATED and cal_results_path.exists():
        results_ours = load_pkl(cal_results_path)
        calibration_report = load_pkl(Path(CFG.snn.SNN_DIR) / "calibration_report.pkl")
    else:
        snn_cal, calibration_report = convert_with_calibration(
            cnn,
            loaders["val"],
            device,
            target_firing_rate=CFG.snn.TARGET_FIRING_RATE,
            calib_batches=CFG.snn.CALIB_BATCHES,
            verbose=True,
        )
        results_ours = sweep_T_values(snn_cal, loaders["test"], device, CFG.snn.T_VALUES, label="calibrated")
        save_pkl(results_ours, cal_results_path)
        save_pkl(calibration_report, Path(CFG.snn.SNN_DIR) / "calibration_report.pkl")

    energies_std = compute_snn_energy_from_results(results_std, cost_model)
    energies_ours = compute_snn_energy_from_results(results_ours, cost_model)
    print_comparison_table(cnn_energy, cnn_metrics, results_std, results_ours, energies_std, energies_ours)

    dump_results_summary(
        CFG.snn.STD_RESULTS_JSON,
        {"results": results_std, "energies": energies_std},
    )
    dump_results_summary(
        CFG.snn.CAL_RESULTS_JSON,
        {"results": results_ours, "energies": energies_ours, "calibration_report": calibration_report},
    )

    if not SKIP_FIGURES:
        if Path(CFG.cnn.LOG_CSV).exists():
            plot_training_curves(CFG.cnn.LOG_CSV)
        best_T = max(results_std, key=lambda t: results_std[t]["f1_macro"])
        plot_confusion_matrices(cnn_metrics["confusion_matrix"], results_std[best_T]["confusion_matrix"], best_T)
        plot_accuracy_vs_T(results_std, results_ours, cnn_metrics["accuracy"], cnn_metrics["f1_macro"])
        plot_pareto_curve(cnn_energy, cnn_metrics, energies_std, results_std, energies_ours, results_ours)
        plot_spike_rate_heatmap(energies_ours, CFG.snn.T_VALUES)
        plot_ablation_table(results_std, results_ours, energies_std, energies_ours, cnn_energy)

    best_std_T = max(results_std, key=lambda t: results_std[t]["f1_macro"])
    best_std_result = results_std[best_std_T]
    best_std_energy = energies_std[best_std_T]
    ratio_std = cnn_energy["energy_uJ"] / best_std_energy["energy_uJ"] if best_std_energy["energy_uJ"] > 0 else 0.0

    best_cal_T = max(results_ours, key=lambda t: results_ours[t]["f1_macro"])
    best_cal_result = results_ours[best_cal_T]

    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    print(f"CNN         : acc={cnn_metrics['accuracy']*100:.2f}% f1={cnn_metrics['f1_macro']*100:.2f}%")
    print(
        f"Standard SNN: acc={best_std_result['accuracy']*100:.2f}% "
        f"f1={best_std_result['f1_macro']*100:.2f}% T={best_std_T} "
        f"ratio={ratio_std:.1f}x"
    )
    print(
        f"Calib. SNN  : acc={best_cal_result['accuracy']*100:.2f}% "
        f"f1={best_cal_result['f1_macro']*100:.2f}% T={best_cal_T}"
    )


if __name__ == "__main__":
    main()
