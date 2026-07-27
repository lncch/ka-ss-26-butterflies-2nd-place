#!/usr/bin/env python3
"""Rebuild the exact submission that scored public 0.9183 / private 0.9075.

This is NOT a single notebook run. The scoring submission averaged four model outputs produced
in four separate training runs, two of which were then re-inferred at 256px:

    convnext_base.fb_in22k_ft_in1k  seed 61, 18 epochs   -> re-inferred at 256px  (FixRes)
    convnext_base.fb_in22k_ft_in1k  seed 11, 12 epochs   -> re-inferred at 256px  (FixRes)
    deit3_base_patch16_224          seed 51, 12 epochs   -> kernel output, 224px
    swin_small_patch4_window7_224   seed  7, 12 epochs   -> kernel output, 224px

All four: FOLDS=1 (train on 80%), grayscale-matched, hflip TTA, equal weight.

Steps to reproduce from scratch:
  1. Train the four models. Each is notebooks/train_gray.ipynb with ENSEMBLE_MODELS set to the one
     backbone and SEED/EPOCHS as above. Collect each run's test_probs_fold0.npy and fold0_raw.pt.
  2. Re-infer the two convnext models at 256px with scripts/fixres.py. FixRes is worth checking
     rather than assuming -- it helped one of these two and hurt the other on their respective
     holdouts, so it is not a reliable gain.
  3. Run this script to average the four and write the submission.

Note the honest caveat: step 2 is why this beat the plain 224px blend on the leaderboard, but its
own validation was inconsistent. See README.md section 7.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"

MEMBERS = [
    ("convnext_base@18ep 256px", ROOT / "artifacts/probs/cb18_256_test.npy"),
    ("convnext_base@12ep 256px", ROOT / "artifacts/probs/cb12_256_test.npy"),
    ("deit3_base 224px", ROOT / "artifacts/probs/deit3_224_test.npy"),
    ("swin_small 224px", ROOT / "artifacts/probs/swin_small_224_test.npy"),
]


def load(p: Path):
    if p.is_file():
        a = np.load(p)
    elif p.is_dir():
        f = sorted(p.rglob("test_probs_fold*.npy"))
        if not f:
            return None
        a = np.mean([np.load(x) for x in f], axis=0)
    else:
        return None
    return a / a.sum(1, keepdims=True)


def main() -> None:
    probs, names = [], []
    for name, path in MEMBERS:
        a = load(path)
        if a is None:
            print(f"  MISSING {name}  ({path})")
            continue
        probs.append(a); names.append(name)
        print(f"  loaded  {name}")
    if len(probs) != len(MEMBERS):
        raise SystemExit(f"only {len(probs)}/{len(MEMBERS)} members available; cannot reproduce exactly")

    blend = np.mean(probs, axis=0)
    classes = sorted(pd.read_csv(DATA / "train.csv")["label"].unique())
    te = pd.read_csv(DATA / "test.csv")
    out = ROOT / "reproduced_best.csv"
    pd.DataFrame({"path": te["path"],
                  "label": [classes[i] for i in blend.argmax(1)]}).to_csv(out, index=False)
    print(f"\nwrote {out} from {len(probs)} members (expected public 0.9183 / private 0.9075)")


if __name__ == "__main__":
    main()
