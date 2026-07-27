#!/usr/bin/env python3
"""Push a configured copy of the notebook to Kaggle as a headless GPU run.

Lets us run several backbones concurrently instead of one interactive session at a time.
Each kernel writes its own submission.csv and test_probs, which we download and blend locally.

usage: push_kernel.py <slug> <model> <folds> <epochs> <budget_min> [seed]
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "almoayyadabuljdail"
NB = ROOT / "notebooks/train_gray.ipynb"


def main() -> None:
    slug, model, folds, epochs, budget = sys.argv[1:6]
    seed = sys.argv[6] if len(sys.argv) > 6 else "42"

    nb = json.loads(NB.read_text())
    cfg = next(c for c in nb["cells"]
               if c["cell_type"] == "code" and "ENSEMBLE_MODELS" in "".join(c["source"]))
    cfg["source"] = [
        "# CONFIG — patched by push_kernel.py for a headless run\n",
        f"ENSEMBLE_MODELS = ['{model}']\n",
        "IMG_SIZE = 224\n",
        f"EPOCHS = {epochs}\n",
        f"FOLDS = {folds}\n",
        "RUN_MODE = 'full'\n",
        f"TIME_BUDGET_MIN = {budget}\n",
        "USE_EMA = False\n",
        "P_LUMA_JITTER = 0.5\n",
        "MATCH_JPEG = True\n",
        "VERTICAL_FLIP = False\n",
        "BATCH_SIZE = 64\n",
        "NUM_WORKERS = 4\n",
        f"SEED = {seed}\n",
        "DATA_ROOT = '/kaggle/input/competitions/ka-ss-26-challenge-1'\n",
        "OUTPUT_DIR = '/kaggle/working'\n",
    ]
    # SEED is a Config field but was not previously threaded through the run cell.
    run = next(c for c in nb["cells"] if "run_ensemble(cfg" in "".join(c["source"]))
    run["source"] = [s.replace("NUM_WORKERS=NUM_WORKERS,", "NUM_WORKERS=NUM_WORKERS, SEED=SEED,")
                     for s in run["source"]]

    out = ROOT / "kernels" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_gray.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{slug}",
        "title": slug,
        "code_file": "train_gray.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "competition_sources": ["ka-ss-26-challenge-1"],
        "dataset_sources": [],
        "kernel_sources": [],
        # Case-sensitive. "nvidiaTeslaT4" is silently ignored and you get a P100,
        # which is broken on Kaggle (sm_60 vs a sm_70+ torch build).
        "machine_shape": "NvidiaTeslaT4",
    }, indent=2) + "\n")

    print(f"pushing {slug}: {model} folds={folds} epochs={epochs} budget={budget}min seed={seed}")
    r = subprocess.run(["kaggle", "kernels", "push", "-p", str(out), "--accelerator", "NvidiaTeslaT4"],
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())


if __name__ == "__main__":
    main()
