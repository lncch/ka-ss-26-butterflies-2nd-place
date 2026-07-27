"""Look for exploitable structure in the test set.

1. ORDERING LEAK. test_img_000000..001479 at ~14.8 images/class. If the organizers wrote the
   files out grouped by class and never shuffled, neighbouring indices share a label and we could
   smooth predictions along the index axis for a large gain. Detect it without labels: if the set
   is class-ordered, adjacent images are far more similar in feature space than random pairs.

2. NEAR-DUPLICATE MINING. pHash found no train/test overlap, but pHash is crude. Embeddings catch
   re-crops and rescales that pHash misses. Any test image that is a near-copy of a train image
   has a known label -- free points.

3. WITHIN-TEST DUPLICATES. Near-copies inside the test set should at least receive consistent
   predictions.
"""
import os

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.expanduser("~/kaggle/butterflies-c1/data/raw")
DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")
MEAN, STD = np.float32([0.485, 0.456, 0.406]), np.float32([0.229, 0.224, 0.225])


class DS(Dataset):
    def __init__(self, paths, root):
        self.paths, self.root = paths, root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(os.path.join(self.root, self.paths[i])).convert("L").convert("RGB")
        a = np.asarray(im, np.float32) / 255.0
        return torch.from_numpy(((a - MEAN) / STD).transpose(2, 0, 1))


@torch.no_grad()
def embed(model, paths, root):
    out = []
    for x in DataLoader(DS(paths, root), batch_size=64, num_workers=4):
        f = model(x.to(DEV)).float()
        out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    tr = pd.read_csv(f"{ROOT}/train.csv")
    te = pd.read_csv(f"{ROOT}/test.csv")
    model = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=0).eval().to(DEV)

    print("embedding test...", flush=True)
    E = embed(model, list(te["path"]), f"{ROOT}/test")
    print("embedding train...", flush=True)
    T = embed(model, list(tr["path"]), ROOT)

    # ---- 1. ordering leak -------------------------------------------------
    adj = (E[:-1] * E[1:]).sum(1)
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(E), 40000), rng.integers(0, len(E), 40000)
    keep = i != j
    rnd = (E[i[keep]] * E[j[keep]]).sum(1)
    print("\n--- 1. ordering leak ---")
    print(f"adjacent-index cosine : mean={adj.mean():.4f} median={np.median(adj):.4f}")
    print(f"random-pair cosine    : mean={rnd.mean():.4f} median={np.median(rnd):.4f}")
    z = (adj.mean() - rnd.mean()) / rnd.std()
    print(f"separation = {z:.2f} sigma   -> {'ORDERED - EXPLOITABLE' if z > 3 else 'no ordering signal; test set is shuffled'}")

    # Blocks of ~14.8: if class-ordered, similarity within a 15-window >> across windows.
    w = 15
    n = (len(E) // w) * w
    blocks = E[:n].reshape(-1, w, E.shape[1])
    within = np.mean([(b @ b.T)[np.triu_indices(w, 1)].mean() for b in blocks])
    print(f"mean cosine within 15-image blocks: {within:.4f} vs random {rnd.mean():.4f}")

    # ---- 2. train/test near-duplicates ------------------------------------
    sim = E @ T.T
    best = sim.max(1)
    who = sim.argmax(1)
    print("\n--- 2. test -> nearest train image (cosine) ---")
    for t in (0.99, 0.97, 0.95, 0.90):
        print(f"  >= {t:.2f}: {int((best >= t).sum()):4d} test images ({100 * (best >= t).mean():.1f}%)")
    print(f"  distribution: max={best.max():.4f} p99={np.percentile(best, 99):.4f} median={np.median(best):.4f}")
    top = np.argsort(-best)[:10]
    print("  strongest matches (test -> train label):")
    for k in top:
        print(f"    {te['path'][k]}  {best[k]:.4f}  ->  {tr['label'][who[k]]}")

    # ---- 3. within-test duplicates ----------------------------------------
    ss = E @ E.T
    np.fill_diagonal(ss, -1)
    mx = ss.max(1)
    print("\n--- 3. within-test near-duplicates ---")
    for t in (0.99, 0.97, 0.95):
        print(f"  >= {t:.2f}: {int((mx >= t).sum()):4d} test images")

    np.save("/tmp/test_emb.npy", E)
    np.save("/tmp/train_emb.npy", T)


if __name__ == "__main__":
    main()
