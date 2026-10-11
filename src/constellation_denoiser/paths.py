"""Canonical paths for source data, experiment artifacts, and checkpoints."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARCHIVE_DATA_DIR = RAW_DATA_DIR / "archive"
MANIFEST_DIR = DATA_DIR / "manifest"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
PAPER_DIR = PROJECT_ROOT / "paper"

E001_DIR = EXPERIMENTS_DIR / "E001_pilot_models"
E002_DIR = EXPERIMENTS_DIR / "E002_tanh_complex_paper"
E003_DIR = EXPERIMENTS_DIR / "E003_large_scale_pilot"
E004_DIR = EXPERIMENTS_DIR / "E004_training_file_curve"
E005_DIR = EXPERIMENTS_DIR / "E005_equal_steps"
E006_DIR = EXPERIMENTS_DIR / "E006_plateau_seeds"
E007_DIR = EXPERIMENTS_DIR / "E007_window_width_sweep"
E008_DIR = EXPERIMENTS_DIR / "E008_cosine_lr_coefficients"
E009_DIR = EXPERIMENTS_DIR / "E009_kerr_ablation"

E001_RESULTS = E001_DIR / "results"
E002_RESULTS = E002_DIR / "results"
E003_RESULTS = E003_DIR / "results"
E004_RESULTS = E004_DIR / "results"
E005_RESULTS = E005_DIR / "results"
E006_RESULTS = E006_DIR / "results"
E007_RESULTS = E007_DIR / "results"
E008_RESULTS = E008_DIR / "results"
E009_RESULTS = E009_DIR / "results"

E001_CHECKPOINTS = CHECKPOINTS_DIR / "E001"
E002_CHECKPOINTS = CHECKPOINTS_DIR / "E002"
E003_CHECKPOINTS = CHECKPOINTS_DIR / "E003"
E004_CHECKPOINTS = CHECKPOINTS_DIR / "E004"
E005_CHECKPOINTS = CHECKPOINTS_DIR / "E005"
E006_CHECKPOINTS = CHECKPOINTS_DIR / "E006"
E007_CHECKPOINTS = CHECKPOINTS_DIR / "E007"
E008_CHECKPOINTS = CHECKPOINTS_DIR / "E008"
E009_CHECKPOINTS = CHECKPOINTS_DIR / "E009"


def project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved
