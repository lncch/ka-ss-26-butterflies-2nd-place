#!/usr/bin/env python3
"""FixRes: test at a higher resolution than we trained at, validated on each model's own holdout.

RandomResizedCrop(224, scale=(0.65,1.0)) magnifies objects during training relative to a plain
resize at test time, so the apparent-object-size distributions do not match. Evaluating at a
higher resolution restores the match and usually gains accuracy for free -- no retraining.

We know each model's exact holdout (FOLDS=1 -> train_test_split(random_state=SEED)), so this is
measured on real labels, not assumed. Paired comparison on identical images, so the difference
between resolutions is far better resolved than either absolute score.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")
MEAN = STD = None

MODELS = [
    ("convnext_base.fb_in22k_ft_in1k", "kernel_out/bfly-cnxt-base/model0_convnext_base/fold0_raw.pt", 11),
    ("swin_small_patch4_window7_224.ms_in22k_ft_in1k",
     "kernel_out/bfly-swin-s/model0_swin_small_patch4_window7_224/fold0_raw.pt", 7),
    ("deit3_base_patch16_224.fb_in22k_ft_in1k",
     "kernel_out/bflydeit3/model0_deit3_base_patch16_224/fold0_raw.pt", 51),
]


def gray_array(paths, root, match_jpeg=True):
    out = np.empty((len(paths), 224, 224), np.uint8)
    for i, rel in enumerate(paths):
        g = Image.open(Path(root) / rel).convert("RGB").convert("L")
        if match_jpeg:
            b = io.BytesIO(); g.save(b, "JPEG", quality=95); b.seek(0)
            g = Image.open(b).convert("L")
        out[i] = np.asarray(g, np.uint8)
    return out


class DS(Dataset):
    def __init__(self, arr, size):
        self.arr, self.size = arr, size

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        im = Image.fromarray(self.arr[i]).convert("RGB")
        if self.size != 224:
            im = im.resize((self.size, self.size), Image.BICUBIC)
        a = np.asarray(im, np.float32) / 255.0
        return torch.from_numpy(((a - 0.449) / 0.226).transpose(2, 0, 1))


@torch.inference_mode()
def predict(model, arr, size, bs=32):
    out = []
    for x in DataLoader(DS(arr, size), batch_size=bs, num_workers=2):
        x = x.to(DEV)
        p = model(x).softmax(1) + model(torch.flip(x, (-1,))).softmax(1)
        out.append((p / 2).float().cpu().numpy())
    return np.concatenate(out)


def main():
    sizes = [int(s) for s in (sys.argv[1:] or [224, 256, 288])]
    tr = pd.read_csv(DATA / "train.csv"); te = pd.read_csv(DATA / "test.csv")
    classes = sorted(tr["label"].unique())
    y = tr["label"].map({c: i for i, c in enumerate(classes)}).values

    print("caching grayscale arrays...", flush=True)
    tr_arr = gray_array(tr["path"], DATA)
    te_arr = gray_array(te["path"], DATA / "test")

    for name, ckpt, seed in MODELS:
        _, va = train_test_split(np.arange(len(tr)), test_size=1200, stratify=y, random_state=seed)
        m = timm.create_model(name, pretrained=False, num_classes=100)
        m.load_state_dict(torch.load(ROOT / ckpt, map_location="cpu"))
        m.eval().to(DEV)
        print(f"\n=== {name.split('.')[0]}  (holdout seed {seed}, {len(va)} images) ===", flush=True)
        best, probs_at = None, {}
        for s in sizes:
            p = predict(m, tr_arr[va], s)
            sc = f1_score(y[va], p.argmax(1), average="macro")
            probs_at[s] = sc
            tag = ""
            if best is None or sc > best[1]:
                best = (s, sc); tag = "  <-- best"
            print(f"  val @ {s}px  macro-F1 = {sc:.5f}{tag}", flush=True)
        base = probs_at[224]
        print(f"  FixRes gain at {best[0]}px: {best[1] - base:+.5f}")
        np.save(f"/tmp/fixres_{name.split('.')[0]}_test.npy", predict(m, te_arr, best[0]))
        np.save(f"/tmp/fixres_{name.split('.')[0]}_size.npy", np.array([best[0], best[1]]))
        del m
        if DEV == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
