"""
Research-grade configuration for Parcel3D ANN-to-SNN experiments.

This config is designed for Kaggle background runs and local reproducibility.
Values can be overridden through environment variables so the notebook can
import the code dataset directly without rewriting files.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _env_list_int(name: str, default: List[int]) -> List[int]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _default_output_dir() -> str:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return str(kaggle_working / "outputs")
    return str((Path.cwd() / "artifacts" / "outputs").resolve())


@dataclass
class DataConfig:
    DATA_ROOT: str = _env_str(
        "PARCEL3D_DATA_ROOT",
        "/kaggle/input/datasets/ayesha19765/parcel3d/parcel3d",
    )
    OUTPUT_DIR: str = _env_str("PARCEL3D_OUTPUT_DIR", _default_output_dir())

    IMG_SIZE: int = _env_int("PARCEL3D_IMG_SIZE", 224)
    BATCH_SIZE: int = _env_int("PARCEL3D_BATCH_SIZE", 32)
    NUM_WORKERS: int = _env_int("PARCEL3D_NUM_WORKERS", 0)
    PIN_MEMORY: bool = _env_bool("PARCEL3D_PIN_MEMORY", True)

    MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    CACHE_DIR: str = field(init=False)

    def __post_init__(self) -> None:
        self.CACHE_DIR = f"{self.OUTPUT_DIR}/cache"


@dataclass
class CNNConfig:
    SEEDS: List[int] = field(
        default_factory=lambda: _env_list_int("PARCEL3D_SEEDS", [42, 123, 456])
    )

    STAGE1_EPOCHS: int = _env_int("PARCEL3D_STAGE1_EPOCHS", 5)
    STAGE1_LR: float = _env_float("PARCEL3D_STAGE1_LR", 1e-3)
    STAGE1_WD: float = _env_float("PARCEL3D_STAGE1_WD", 1e-4)

    STAGE2_EPOCHS: int = _env_int("PARCEL3D_STAGE2_EPOCHS", 12)
    STAGE2_LR: float = _env_float("PARCEL3D_STAGE2_LR", 5e-5)
    STAGE2_WD: float = _env_float("PARCEL3D_STAGE2_WD", 1e-4)

    PATIENCE: int = _env_int("PARCEL3D_PATIENCE", 4)
    ETA_MIN: float = _env_float("PARCEL3D_ETA_MIN", 1e-6)
    DROPOUT: float = _env_float("PARCEL3D_DROPOUT", 0.30)
    LABEL_SMOOTHING: float = _env_float("PARCEL3D_LABEL_SMOOTHING", 0.05)
    USE_AMP: bool = _env_bool("PARCEL3D_USE_AMP", True)

    CHECKPOINT_DIR: str = field(init=False)
    LOG_CSV: str = field(init=False)
    METRICS_JSON: str = field(init=False)

    def __post_init__(self) -> None:
        output_dir = _env_str("PARCEL3D_OUTPUT_DIR", _default_output_dir())
        self.CHECKPOINT_DIR = f"{output_dir}/cnn"
        self.LOG_CSV = f"{self.CHECKPOINT_DIR}/training_log_all_seeds.csv"
        self.METRICS_JSON = f"{self.CHECKPOINT_DIR}/cnn_test_metrics.json"


@dataclass
class SNNConfig:
    T_VALUES: List[int] = field(
        default_factory=lambda: _env_list_int("PARCEL3D_T_VALUES", [32, 64, 128, 256])
    )
    STANDARD_PERCENTILE: float = _env_float("PARCEL3D_STANDARD_PERCENTILE", 99.9)
    TARGET_FIRING_RATE: float = _env_float("PARCEL3D_TARGET_FIRING_RATE", 0.10)
    CALIB_BATCHES: int = _env_int("PARCEL3D_CALIB_BATCHES", 8)
    MIN_SCALE: float = _env_float("PARCEL3D_CALIB_MIN_SCALE", 0.70)
    MAX_SCALE: float = _env_float("PARCEL3D_CALIB_MAX_SCALE", 1.30)

    SNN_DIR: str = field(init=False)
    STD_RESULTS_JSON: str = field(init=False)
    CAL_RESULTS_JSON: str = field(init=False)

    def __post_init__(self) -> None:
        output_dir = _env_str("PARCEL3D_OUTPUT_DIR", _default_output_dir())
        self.SNN_DIR = f"{output_dir}/snn"
        self.STD_RESULTS_JSON = f"{output_dir}/results_standard_summary.json"
        self.CAL_RESULTS_JSON = f"{output_dir}/results_calibrated_summary.json"


@dataclass
class EnergyConfig:
    E_MAC_PJ: float = _env_float("PARCEL3D_E_MAC_PJ", 4.6)
    E_AC_PJ: float = _env_float("PARCEL3D_E_AC_PJ", 0.9)


@dataclass
class ExperimentConfig:
    NUM_CLASSES: int = 2
    CLASS_NAMES: List[str] = field(default_factory=lambda: ["normal", "damaged"])

    data: DataConfig = field(default_factory=DataConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    snn: SNNConfig = field(default_factory=SNNConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)

    FIG_DIR: str = field(init=False)
    RUN_METADATA_JSON: str = field(init=False)

    def __post_init__(self) -> None:
        self.FIG_DIR = f"{self.data.OUTPUT_DIR}/figures"
        self.RUN_METADATA_JSON = f"{self.data.OUTPUT_DIR}/run_metadata.json"
        for path in [
            self.data.OUTPUT_DIR,
            self.data.CACHE_DIR,
            self.cnn.CHECKPOINT_DIR,
            self.snn.SNN_DIR,
            self.FIG_DIR,
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def dump_metadata(self) -> None:
        payload = asdict(self)
        with open(self.RUN_METADATA_JSON, "w") as f:
            json.dump(payload, f, indent=2)


CFG = ExperimentConfig()
