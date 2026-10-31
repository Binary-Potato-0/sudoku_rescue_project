"""Project-wide constants. CHANGE TEAM_SEED to your own team's seed (see README)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DEMO_OUT = ROOT / "outputs" / "demo"

TEAM_SEED = 2026          # every team must use its own generator seed
SPLIT_SEED = 0            # train/validation split of MNIST (kept fixed so all teams share it)
MNIST_FILE = DATA_PROCESSED / "mnist19.npz"

# Reading rule: a 28x28 cell whose mean brightness is below this is BLANK (not decided by the NN)
BLANK_MEAN_THRESHOLD = 0.02
