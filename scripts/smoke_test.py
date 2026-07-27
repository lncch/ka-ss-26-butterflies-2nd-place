#!/usr/bin/env python3
"""Run the real-data CPU smoke test and validate its submission."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.pipeline import Config, run, run_ensemble, sanity_check_grayscale


def main() -> None:
    data = ROOT / "data" / "raw"
    out = ROOT / "artifacts" / "smoke"
    sanity_check_grayscale(data, samples=32)
    path = run(Config(
        DATA_ROOT=str(data), OUTPUT_DIR=str(out), RUN_MODE="smoke",
        TIME_BUDGET_MIN=3, USE_MPS=False,
    ))
    submission = pd.read_csv(path)
    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")
    assert submission.shape == (1480, 2), submission.shape
    assert submission.columns.tolist() == ["path", "label"]
    assert submission["path"].tolist() == test["path"].tolist()
    assert set(submission["label"]) <= set(train["label"].unique())

    # Exercise the multi-backbone path too (smoke mode forces one tiny model, so this
    # validates the ensemble plumbing rather than the architectures).
    ens = run_ensemble(Config(
        DATA_ROOT=str(data), OUTPUT_DIR=str(out / "ens"), RUN_MODE="smoke",
        TIME_BUDGET_MIN=30, USE_MPS=False,
    ), ["mobilenetv3_small_050", "mobilenetv3_small_050"])
    e = pd.read_csv(ens)
    assert e.shape == (1480, 2), e.shape
    assert e["path"].tolist() == test["path"].tolist()
    # Both backbones must actually have run, or the ensemble maths went untested.
    assert (out / "ens" / "ensemble_test_probs.npy").exists()
    assert len(list((out / "ens").glob("model*_*"))) == 2, "second backbone was skipped"
    print("ENSEMBLE SMOKE PASSED: run_ensemble produced a valid 1480-row submission.")

    print("SMOKE TEST PASSED: valid 1480-row submission produced on CPU.")


if __name__ == "__main__":
    main()
