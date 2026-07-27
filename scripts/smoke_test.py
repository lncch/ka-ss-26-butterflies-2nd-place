#!/usr/bin/env python3
"""Run the real-data CPU smoke test and validate its submission."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.pipeline import Config, run, run_solution, sanity_check_grayscale


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

    # Exercise the full solution path: several members, each at its own inference size.
    ens = run_solution(Config(
        DATA_ROOT=str(data), OUTPUT_DIR=str(out / "sol"), RUN_MODE="smoke",
        TIME_BUDGET_MIN=30, USE_MPS=False,
    ), [{"model": "mobilenetv3_small_050", "epochs": 1, "seed": 1, "infer_size": 64},
        {"model": "mobilenetv3_small_050", "epochs": 1, "seed": 2, "infer_size": 80}])
    e = pd.read_csv(ens)
    assert e.shape == (1480, 2), e.shape
    assert e["path"].tolist() == test["path"].tolist()
    assert set(e["label"]) <= set(train["label"].unique())
    assert len(list((out / "sol").glob("member*"))) == 2, "second member did not run"
    assert (out / "sol" / "ensemble_test_probs.npy").exists()
    print("SOLUTION SMOKE PASSED: run_solution blended all members into a valid submission.")

    print("SMOKE TEST PASSED: valid 1480-row submission produced on CPU.")


if __name__ == "__main__":
    main()
