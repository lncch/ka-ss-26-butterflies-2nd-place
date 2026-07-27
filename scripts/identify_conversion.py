"""Which RGB->grayscale conversion did the organizers actually use?

We guessed PIL's ITU-R 601-2 luma. If they used something else, every training image is
slightly off-distribution from every test image, which would show up as a residual OOF->LB gap.

Train and test are both butterfly photos drawn from the same source dataset, so their aggregate
pixel statistics should agree *if and only if* we apply the same conversion they did. Convert the
train set under each candidate weighting and score it against the real test set by
Wasserstein distance between aggregate grayscale histograms (plus a gradient-energy check,
which is sensitive to the weighting in a different way).
"""
import io
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wasserstein_distance

ROOT = os.path.expanduser("~/kaggle/butterflies-c1/data/raw")
N = 1480  # match the test set size so histograms are comparable

CANDIDATES = {
    "ITU-R 601-2 (PIL convert L, cv2)": (0.299, 0.587, 0.114),
    "ITU-R 709 (sRGB luma)":            (0.2126, 0.7152, 0.0722),
    "flat channel average":             (1 / 3, 1 / 3, 1 / 3),
    "601 w/ gamma-correct linear":      None,   # handled specially
    "green channel only":               (0.0, 1.0, 0.0),
}


def to_gray(rgb: np.ndarray, w) -> np.ndarray:
    if w is None:                                     # linear-light luminance, then re-encode gamma
        lin = np.where(rgb / 255 <= 0.04045, rgb / 255 / 12.92, ((rgb / 255 + 0.055) / 1.055) ** 2.4)
        y = lin @ np.array([0.2126, 0.7152, 0.0722])
        y = np.where(y <= 0.0031308, y * 12.92, 1.055 * y ** (1 / 2.4) - 0.055)
        return np.clip(y * 255, 0, 255).astype(np.uint8)
    return np.clip(rgb @ np.array(w), 0, 255).astype(np.uint8)


def jpeg_roundtrip(a: np.ndarray) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=95)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"))


def grad_energy(a: np.ndarray) -> float:
    a = a.astype(np.float32)
    return float(np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean())


def main() -> None:
    tr = pd.read_csv(f"{ROOT}/train.csv").sample(N, random_state=0)
    te = pd.read_csv(f"{ROOT}/test.csv")

    test_px, test_ge = [], []
    for p in te["path"]:
        a = np.asarray(Image.open(f"{ROOT}/test/{p}").convert("L"))
        test_px.append(a.ravel()[::37])
        test_ge.append(grad_energy(a))
    test_px = np.concatenate(test_px)
    print(f"test: {len(te)} images, mean={test_px.mean():.2f} std={test_px.std():.2f} "
          f"grad_energy={np.mean(test_ge):.3f}\n")

    rgbs = [np.asarray(Image.open(f"{ROOT}/{p}").convert("RGB"), dtype=np.float32) for p in tr["path"]]

    print(f"{'conversion':36s} {'wasserstein':>12s} {'mean':>8s} {'std':>8s} {'grad_en':>9s}")
    print("-" * 78)
    results = []
    for name, w in CANDIDATES.items():
        px, ge = [], []
        for rgb in rgbs:
            g = jpeg_roundtrip(to_gray(rgb, w))
            px.append(g.ravel()[::37])
            ge.append(grad_energy(g))
        px = np.concatenate(px)
        d = wasserstein_distance(px, test_px)
        results.append((d, name))
        print(f"{name:36s} {d:12.4f} {px.mean():8.2f} {px.std():8.2f} {np.mean(ge):9.3f}")

    results.sort()
    print(f"\nbest match: {results[0][1]}  (wasserstein {results[0][0]:.4f})")
    print(f"we currently use: ITU-R 601-2 (PIL convert L, cv2)")
    spread = results[-1][0] - results[0][0]
    print(f"spread across candidates: {spread:.4f} "
          f"-- if this is small relative to the best distance, the test is not conclusive")


if __name__ == "__main__":
    main()
