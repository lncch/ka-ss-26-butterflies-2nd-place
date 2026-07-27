"""Grayscale-matched training pipeline for KA_SS_26 Challenge 1."""
# %% Imports
from __future__ import annotations

import io
import math
import os
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2


# %% Rule compliance: ImageNet-only weight allowlist
MODEL_ZOO = [
    "convnext_small.fb_in22k_ft_in1k",
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
]
# Rule check: "only generic (e.g. ImageNet) weights are allowed".
# Match on the weight tag timm actually resolves, not on the string the user typed --
# tags carry a vendor/recipe prefix ("fb_in22k_ft_in1k", "ms_in22k_ft_in1k", "a1_in1k"),
# and a bare arch name resolves to a default tag that is never spelled out in the name.
IMAGENET_CORPORA = ("in1k", "in12k", "in21k", "in22k")
NON_IMAGENET_MARKERS = (
    "clip", "laion", "openai", "dinov2", "jft", "inat", "webli", "merged2b", "dfn", "siglip",
)


def assert_imagenet_only(model_name: str) -> str:
    """Verify the checkpoint timm will download was pretrained on ImageNet alone."""
    tag = (timm.get_pretrained_cfg(model_name).tag or "").lower()
    assert any(c in tag for c in IMAGENET_CORPORA), (
        f"{model_name!r}: resolved weight tag {tag!r} does not name an ImageNet corpus."
    )
    found = sorted({m for m in NON_IMAGENET_MARKERS if m in f"{model_name.lower()} {tag}"})
    assert not found, (
        f"{model_name!r}: tag {tag!r} indicates non-ImageNet pretraining data ({', '.join(found)})."
    )
    return tag


# %% Configuration
@dataclass
class Config:
    DATA_ROOT: str = "/kaggle/input/competitions/ka-ss-26-challenge-1"
    OUTPUT_DIR: str = "."
    MODEL_NAME: str = MODEL_ZOO[0]
    IMG_SIZE: int = 224
    # Observed on a real run: val macro-F1 plateaus around epoch 11-12, so 14 captures it.
    EPOCHS: int = 14
    FOLDS: int = 3
    # EMA lost to the raw weights on every fold of a real 18-epoch run: decay 0.999 has a
    # ~1000-step horizon but the whole run is only ~1350 steps, so it never catches up.
    # Off by default; it costs an extra full validation pass per epoch.
    USE_EMA: bool = False
    EMA_DECAY: float = 0.99
    RUN_MODE: str = "full"
    # Kaggle sessions cap at 12 h; set this to the time actually available.
    TIME_BUDGET_MIN: float = 180
    P_LUMA_JITTER: float = 0.5
    MATCH_JPEG: bool = True
    VERTICAL_FLIP: bool = False
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 4
    LR: float = 3e-4
    WEIGHT_DECAY: float = 0.05
    LABEL_SMOOTHING: float = 0.1
    SEED: int = 42
    PRETRAINED: bool = True
    USE_MPS: bool = True
    TTA_SCALES: tuple[float, ...] = (1.0,)


def configure(cfg: Config) -> Config:
    if cfg.RUN_MODE == "smoke":
        return replace(
            cfg, MODEL_NAME="mobilenetv3_small_050", IMG_SIZE=64, EPOCHS=1,
            FOLDS=1, BATCH_SIZE=64, NUM_WORKERS=0, PRETRAINED=False,
            P_LUMA_JITTER=0.5, MATCH_JPEG=False, TTA_SCALES=(1.0,),
        )
    if cfg.RUN_MODE == "quick":
        return replace(cfg, FOLDS=1, EPOCHS=8)
    if cfg.RUN_MODE != "full":
        raise ValueError("RUN_MODE must be 'smoke', 'quick', or 'full'")
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# %% Paths and the grayscale sanity check
def resolve_path(root: Path, rel: str, is_test: bool) -> Path:
    p = root / rel
    if p.exists():
        return p
    if is_test:
        return root / "test" / rel
    return p


def sanity_check_grayscale(root: str | Path, samples: int | None = None) -> None:
    root = Path(root)
    test_df = pd.read_csv(root / "test.csv")
    rows = test_df if samples is None else test_df.sample(min(samples, len(test_df)), random_state=42)
    for rel in rows["path"]:
        a = np.asarray(Image.open(resolve_path(root, rel, True)).convert("RGB"))
        assert np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2]), rel
    print(f"Sanity: {len(rows)}/{len(rows)} checked test images are exactly grayscale.")


# %% Dataset: grayscale matching + augmentation
class ButterflyDataset(Dataset):
    """Always emits 3 identical grayscale channels."""

    def __init__(
        self, frame: pd.DataFrame, root: str | Path, labels: dict[str, int] | None,
        train: bool, img_size: int, p_luma_jitter: float = 0.0,
        match_jpeg: bool = False, vertical_flip: bool = False,
        gray_images: np.ndarray | None = None,
    ):
        self.frame, self.root, self.labels = frame.reset_index(drop=True), Path(root), labels
        self.train, self.p_luma_jitter, self.match_jpeg = train, p_luma_jitter, match_jpeg
        self.gray_images = gray_images
        ops: list[object] = []
        if train:
            ops += [
                transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
            ]
            if vertical_flip:
                ops.append(transforms.RandomVerticalFlip(0.5))
            ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
        else:
            ops += [transforms.Resize((img_size, img_size))]
        ops += [
            transforms.ToTensor(),
            # Shared statistics retain the invariant that all three channels match.
            transforms.Normalize((0.449, 0.449, 0.449), (0.226, 0.226, 0.226)),
        ]
        if train:
            ops.append(transforms.RandomErasing(p=0.25))
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.frame)

    def _image(self, index: int, path: Path) -> Image.Image:
        if self.train and self.p_luma_jitter > 0 and random.random() < self.p_luma_jitter:
            rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            weights = np.array([0.299, 0.587, 0.114]) + np.random.uniform(-0.05, 0.05, 3)
            weights = np.clip(weights, 1e-4, None)
            weights /= weights.sum()
            # 2-D uint8 already implies mode "L"; passing it is deprecated in Pillow 13.
            gray = Image.fromarray(np.clip(rgb @ weights, 0, 255).astype(np.uint8))
            if self.match_jpeg:
                buf = io.BytesIO()
                gray.save(buf, "JPEG", quality=95)
                buf.seek(0)
                gray = Image.open(buf).convert("L").copy()
        else:
            if self.gray_images is None:
                gray = Image.open(path).convert("RGB").convert("L")
            else:
                gray = Image.fromarray(self.gray_images[index])
        return gray.convert("RGB")

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        is_test = self.labels is None
        image = self.transform(self._image(index, resolve_path(self.root, row["path"], is_test)))
        if is_test:
            return image
        return image, self.labels[row["label"]]


# %% Pre-decode grayscale once, shared across workers
def predecode_grayscale(
    frame: pd.DataFrame, root: str | Path, img_size: int, match_jpeg: bool, is_test: bool
) -> np.ndarray:
    """Decode standard grayscale images once into a worker-shared uint8 array."""
    root = Path(root)
    images = np.empty((len(frame), img_size, img_size), dtype=np.uint8)
    for i, rel in enumerate(frame["path"]):
        gray = Image.open(resolve_path(root, rel, is_test)).convert("RGB").convert("L")
        if match_jpeg:
            buf = io.BytesIO()
            gray.save(buf, "JPEG", quality=95)
            buf.seek(0)
            gray = Image.open(buf).convert("L").copy()
        gray = gray.resize((img_size, img_size), Image.Resampling.BILINEAR)
        images[i] = np.asarray(gray, dtype=np.uint8)
    return images


# %% Model, device, loaders, LR schedule
def make_model(cfg: Config, n_classes: int) -> nn.Module:
    if cfg.PRETRAINED:
        tag = assert_imagenet_only(cfg.MODEL_NAME)
        print(f"Weights OK: {cfg.MODEL_NAME} (tag={tag}, ImageNet-only).")
    return timm.create_model(
        cfg.MODEL_NAME, pretrained=cfg.PRETRAINED, num_classes=n_classes, drop_path_rate=0.1
    )


def device_for(cfg: Config) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if cfg.USE_MPS and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def loader(ds: Dataset, cfg: Config, shuffle: bool, batch_size: int | None = None) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size or cfg.BATCH_SIZE, shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.NUM_WORKERS > 0,
    )


def cosine_lr(optimizer, epoch: int, cfg: Config) -> None:
    warmup = min(2, max(1, cfg.EPOCHS // 4))
    if epoch < warmup:
        factor = (epoch + 1) / warmup
    else:
        progress = (epoch - warmup + 1) / max(1, cfg.EPOCHS - warmup)
        factor = 0.5 * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = cfg.LR * factor


# %% Inference with flip + multi-scale TTA
@torch.inference_mode()
def predict(
    model: nn.Module, dl: DataLoader, device: torch.device, flip: bool = False,
    scales: tuple[float, ...] = (1.0,),
) -> np.ndarray:
    model.eval()
    out = []
    for batch in dl:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device)
        amp = torch.amp.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else torch.autocast(device_type=device.type, enabled=False)
        with amp:
            predictions = []
            for scale in scales:
                size = max(1, round(x.shape[-1] * scale))
                scaled = x if size == x.shape[-1] else torch.nn.functional.interpolate(
                    x, size=(size, size), mode="bilinear", align_corners=False
                )
                predictions.append(model(scaled).softmax(1))
                if flip:
                    predictions.append(model(torch.flip(scaled, (-1,))).softmax(1))
            probs = torch.stack(predictions).mean(0)
        out.append(probs.float().cpu().numpy())
    return np.concatenate(out)


# %% One training epoch
def train_epoch(model, ema, dl, optimizer, scaler, criterion, mixup, device) -> float:
    model.train()
    total, count = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        x, targets = mixup(x, y)
        optimizer.zero_grad(set_to_none=True)
        amp = torch.amp.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else torch.autocast(device_type=device.type, enabled=False)
        with amp:
            loss = criterion(model(x), targets)
        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if ema is not None:
            ema.update(model)
        total += loss.item() * len(x)
        count += len(x)
    return total / count


# %% Macro-F1 prior correction
def tune_prior(probs: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    before = f1_score(y, probs.argmax(1), average="macro")
    marginal = np.clip(probs.mean(0), 1e-8, None)
    best_tau, best_probs, best = 0.0, probs, before
    logp = np.log(np.clip(probs, 1e-12, None))
    for tau in np.linspace(0.05, 1.0, 20):
        z = logp - tau * np.log(marginal)[None]
        adjusted = np.exp(z - z.max(1, keepdims=True))
        adjusted /= adjusted.sum(1, keepdims=True)
        score = f1_score(y, adjusted.argmax(1), average="macro")
        if score > best:
            best_tau, best_probs, best = float(tau), adjusted, score
    if best - before <= 0.0005:
        best_tau, best_probs, best = 0.0, probs, before
    print(f"OOF macro-F1 prior correction: {before:.5f} -> {best:.5f} (tau={best_tau:.2f})")
    return best_tau, best_probs


def apply_prior(probs: np.ndarray, marginal: np.ndarray, tau: float) -> np.ndarray:
    if tau == 0:
        return probs
    z = np.log(np.clip(probs, 1e-12, None)) - tau * np.log(np.clip(marginal, 1e-8, None))[None]
    out = np.exp(z - z.max(1, keepdims=True))
    return out / out.sum(1, keepdims=True)


# %% Fold splits and dataset discovery
def _splits(frame: pd.DataFrame, cfg: Config):
    y = frame["label"]
    if cfg.FOLDS == 1:
        # Smoke has two examples/class, so validation must be at least one/class.
        test_size = max(int(math.ceil(0.2 * len(frame))), y.nunique())
        a, b = train_test_split(
            np.arange(len(frame)), test_size=test_size, stratify=y, random_state=cfg.SEED
        )
        return [(a, b)]
    return list(StratifiedKFold(cfg.FOLDS, shuffle=True, random_state=cfg.SEED).split(frame, y))


def resolve_data_root(configured: str, slug: str = "ka-ss-26-challenge-1") -> Path:
    """Find the dataset wherever Kaggle actually mounted it.

    Attaching the competition as a notebook Input gives /kaggle/input/..., while
    kagglehub.competition_download() returns a cache directory instead. Try the
    configured path first, then the usual mount points, then kagglehub.
    """
    candidates = [Path(configured), Path("/kaggle/input/competitions") / slug,
                  Path("/kaggle/input") / slug, Path("/kaggle/working") / slug, Path(slug)]
    for c in candidates:
        if (c / "train.csv").exists():
            return c
    for csv in sorted(Path("/kaggle/input").glob("*/train.csv")) if Path("/kaggle/input").exists() else []:
        return csv.parent
    try:
        import kagglehub
        got = Path(kagglehub.competition_download(slug))
        if (got / "train.csv").exists():
            return got
        for csv in sorted(got.glob("*/train.csv")):
            return csv.parent
    except Exception as exc:
        print(f"kagglehub fallback unavailable: {exc}")
    raise FileNotFoundError(
        f"No train.csv found. Tried: {[str(c) for c in candidates]}. "
        "Attach the competition under Input, or set DATA_ROOT to the printed kagglehub path."
    )


# %% Train / validate / predict loop
def load_frames(cfg: Config):
    """Root, train/test frames and class list -- shared so every caller sees identical rows.

    run() and run_ensemble() must agree on which training rows exist, or the OOF arrays one
    writes cannot be indexed by the labels the other builds.
    """
    root = resolve_data_root(cfg.DATA_ROOT)
    train_df, test_df = pd.read_csv(root / "train.csv"), pd.read_csv(root / "test.csv")
    classes = sorted(train_df["label"].unique())
    if cfg.RUN_MODE == "smoke":
        train_df = (train_df.groupby("label", sort=True, group_keys=False)
                    .sample(n=2, random_state=cfg.SEED).reset_index(drop=True))
    return root, train_df, test_df, classes


def run(config: Config) -> Path:
    cfg = configure(config)
    seed_everything(cfg.SEED)
    root, train_df, test_df, classes = load_frames(cfg)
    output = Path(cfg.OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    label_to_idx = {v: i for i, v in enumerate(classes)}
    n, nt, nc = len(train_df), len(test_df), len(classes)
    device = device_for(cfg)
    print(f"Mode={cfg.RUN_MODE} device={device} train={n} test={nt} classes={nc}")
    print("Predecoding standard grayscale images.")
    train_gray = predecode_grayscale(train_df, root, cfg.IMG_SIZE, cfg.MATCH_JPEG, False)
    test_gray = predecode_grayscale(test_df, root, cfg.IMG_SIZE, False, True)

    # Acceptance guard: even the normalized/augmented training tensor is 3x gray.
    check_ds = ButterflyDataset(
        train_df.iloc[:1], root, label_to_idx, False, cfg.IMG_SIZE,
        gray_images=train_gray[:1],
    )
    check, _ = check_ds[0]
    assert torch.equal(check[0], check[1]) and torch.equal(check[1], check[2])
    print("Transform sanity: sampled training tensor has three identical channels.")

    oof = np.zeros((n, nc), np.float32)
    covered = np.zeros(n, bool)
    test_folds: list[np.ndarray] = []
    started = time.monotonic()
    fold_durations: list[float] = []
    for fold, (tr_idx, va_idx) in enumerate(_splits(train_df, cfg)):
        elapsed = (time.monotonic() - started) / 60
        if fold_durations and elapsed + float(np.mean(fold_durations)) > cfg.TIME_BUDGET_MIN:
            print("Time budget cannot fit another mean-duration fold; no new fold launched.")
            break
        fold_started = time.monotonic()
        seed_everything(cfg.SEED + fold)
        model = make_model(cfg, nc).to(device)  # deliberately fresh every fold
        print(f"Fold {fold + 1}: fresh model id={id(model)}")
        ema = ModelEmaV2(model, decay=cfg.EMA_DECAY) if cfg.USE_EMA else None
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        criterion = SoftTargetCrossEntropy()
        mixup = Mixup(
            mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5,
            label_smoothing=cfg.LABEL_SMOOTHING, num_classes=nc,
        )
        tr_ds = ButterflyDataset(train_df.iloc[tr_idx], root, label_to_idx, True, cfg.IMG_SIZE,
                                 cfg.P_LUMA_JITTER, cfg.MATCH_JPEG, cfg.VERTICAL_FLIP,
                                 train_gray[tr_idx])
        va_ds = ButterflyDataset(train_df.iloc[va_idx], root, label_to_idx, False, cfg.IMG_SIZE,
                                 0.0, cfg.MATCH_JPEG, gray_images=train_gray[va_idx])
        te_ds = ButterflyDataset(
            test_df, root, None, False, cfg.IMG_SIZE, 0.0, False,
            gray_images=test_gray,
        )
        tr_dl, va_dl, te_dl = loader(tr_ds, cfg, True), loader(va_ds, cfg, False), loader(te_ds, cfg, False)
        y_val = np.array([label_to_idx[x] for x in train_df.iloc[va_idx]["label"]])
        best_score, best_state, best_kind = -1.0, None, "raw"
        for epoch in range(cfg.EPOCHS):
            cosine_lr(optimizer, epoch, cfg)
            try:
                loss = train_epoch(model, ema, tr_dl, optimizer, scaler, criterion, mixup, device)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or cfg.BATCH_SIZE <= 32:
                    raise
                print("OOM at batch 64; retrying fold with batch 32.")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                tr_dl = loader(tr_ds, cfg, True, 32)
                loss = train_epoch(model, ema, tr_dl, optimizer, scaler, criterion, mixup, device)
            raw_f1 = f1_score(y_val, predict(model, va_dl, device).argmax(1), average="macro")
            candidate, kind, score = model, "raw", raw_f1
            extra = ""
            if ema is not None:
                ema_f1 = f1_score(y_val, predict(ema.module, va_dl, device).argmax(1), average="macro")
                extra = f" ema_f1={ema_f1:.4f}"
                if ema_f1 > raw_f1:
                    candidate, kind, score = ema.module, "ema", ema_f1
            if score > best_score:
                best_score, best_kind = score, kind
                best_state = {k: v.detach().cpu().clone() for k, v in candidate.state_dict().items()}
            print(f"fold={fold + 1} epoch={epoch + 1} loss={loss:.4f} raw_f1={raw_f1:.4f}{extra}")
        model.load_state_dict(best_state)
        oof[va_idx] = predict(model, va_dl, device, flip=True, scales=cfg.TTA_SCALES)
        covered[va_idx] = True
        test_folds.append(predict(model, te_dl, device, flip=True, scales=cfg.TTA_SCALES))
        np.save(output / f"test_probs_fold{fold}.npy", test_folds[-1])
        torch.save(best_state, output / f"fold{fold}_{best_kind}.pt")
        fold_durations.append((time.monotonic() - fold_started) / 60)
        print(f"Fold {fold + 1} best={best_score:.5f} ({best_kind})")

        # Always leave a valid submission AND the OOF array behind after every completed
        # fold, so interrupting the run mid-way still leaves everything needed to ensemble.
        np.save(output / "oof_probs.npy", oof)
        np.save(output / "oof_covered.npy", covered)
        current_probs = np.mean(test_folds, axis=0)
        y_all = np.array([label_to_idx[x] for x in train_df["label"]])
        tau, _ = tune_prior(oof[covered], y_all[covered])
        current_probs = apply_prior(current_probs, current_probs.mean(0), tau)
        submission = pd.DataFrame({
            "path": test_df["path"],
            "label": [classes[i] for i in current_probs.argmax(1)],
        })
        submission.to_csv(output / "submission.csv", index=False)

    if not test_folds:
        raise RuntimeError("No fold completed; increase TIME_BUDGET_MIN.")
    np.save(output / "oof_probs.npy", oof)
    test_probs = np.mean(test_folds, axis=0)
    if covered.any():
        y_all = np.array([label_to_idx[x] for x in train_df["label"]])
        tau, _ = tune_prior(oof[covered], y_all[covered])
        test_probs = apply_prior(test_probs, test_probs.mean(0), tau)
    np.save(output / "test_probs.npy", test_probs)
    submission = pd.DataFrame({"path": test_df["path"], "label": [classes[i] for i in test_probs.argmax(1)]})
    submission_path = output / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Wrote {submission_path} with {len(submission)} rows from {len(test_folds)} fold(s).")
    return submission_path


# %% Balanced assignment (Sinkhorn)
def sinkhorn(p: np.ndarray, strength: float, iters: int = 60) -> np.ndarray:
    """Nudge the predicted class marginal toward uniform.

    train is exactly 60/class and 1480 = 80*15 + 20*14, so the test set is almost certainly
    balanced. Under macro-F1 every class counts equally, so a model that starves a class it is
    shy about pays for it. Plain argmax cannot express that constraint; this can.
    """
    if strength <= 0:
        return p
    logp = np.log(np.clip(p, 1e-12, None))
    target = np.log(np.full(p.shape[1], 1.0 / p.shape[1]))
    v = np.zeros(p.shape[1])
    for _ in range(iters):
        z = logp + v
        z -= z.max(1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(1, keepdims=True)
        v += strength * (target - np.log(np.clip(q.mean(0), 1e-12, None)))
    z = logp + v
    z -= z.max(1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(1, keepdims=True)


def tune_sinkhorn(oof: np.ndarray, y: np.ndarray, min_gain: float = 0.0005) -> float:
    """Pick a Sinkhorn strength on OOF. Returns 0 unless it clearly helps.

    The OOF set is balanced exactly like the test set, so this is a fair rehearsal -- but it is
    a small sample, so demand a real margin before changing any prediction.
    """
    base = f1_score(y, oof.argmax(1), average="macro")
    best_s, best = 0.0, base
    for s in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        sc = f1_score(y, sinkhorn(oof, s).argmax(1), average="macro")
        if sc > best:
            best_s, best = s, sc
    if best - base < min_gain:
        print(f"  balanced assignment: no reliable OOF gain ({base:.5f} -> {best:.5f}); keeping argmax")
        return 0.0
    print(f"  balanced assignment: OOF {base:.5f} -> {best:.5f} ({best - base:+.5f}) at strength {best_s}")
    return best_s


# %% Multi-backbone ensemble
def run_ensemble(config: Config, models: "list[str]") -> Path:
    """Train each backbone into its own folder, then average their test probabilities.

    Different architecture families make errors in different places, so averaging them beats
    adding more folds of a single one. Each model's OOF is kept, which means the ensemble's
    macro-F1 can be *measured* rather than assumed -- and the top-level submission.csv is
    rewritten after every model, so stopping early still leaves a valid ensemble behind.
    """
    top = Path(config.OUTPUT_DIR)
    top.mkdir(parents=True, exist_ok=True)
    _, train_df, test_df, classes = load_frames(config)
    y_all = np.array([classes.index(v) for v in train_df["label"]])

    started = time.monotonic()
    oofs, tests, names = [], [], []
    for i, name in enumerate(models):
        used = (time.monotonic() - started) / 60
        remaining = config.TIME_BUDGET_MIN - used
        if i and remaining < 5:
            print(f"Time budget spent ({used:.0f} min); skipping remaining backbones.")
            break
        print(f"\n{'=' * 70}\nBackbone {i + 1}/{len(models)}: {name}  ({remaining:.0f} min left)\n{'=' * 70}")
        sub_dir = top / f"model{i}_{name.split('.')[0]}"
        run(replace(config, MODEL_NAME=name, OUTPUT_DIR=str(sub_dir), TIME_BUDGET_MIN=remaining))

        oofs.append(np.load(sub_dir / "oof_probs.npy"))
        tests.append(np.mean([np.load(f) for f in sorted(sub_dir.glob("test_probs_fold*.npy"))], axis=0))
        names.append(name)

        covered = np.load(sub_dir / "oof_covered.npy")
        for nm, o in zip(names, oofs):
            print(f"  OOF {nm:46s} {f1_score(y_all[covered], o[covered].argmax(1), average='macro'):.5f}")
        if len(oofs) > 1:
            blend = np.mean(oofs, axis=0)
            print(f"  OOF {'ENSEMBLE of ' + str(len(oofs)):46s} "
                  f"{f1_score(y_all[covered], blend[covered].argmax(1), average='macro'):.5f}")
            agree = np.mean([tests[0].argmax(1) == t.argmax(1) for t in tests[1:]])
            print(f"  test-set agreement with backbone 1: {agree:.4f} (0.85-0.93 is healthy diversity)")

        probs = np.mean(tests, axis=0)
        np.save(top / "ensemble_test_probs.npy", probs)
        # Rehearse the balanced-assignment decision rule on OOF, which is balanced exactly like
        # the test set, and only adopt it if it genuinely helps there.
        s = tune_sinkhorn(np.mean(oofs, axis=0)[covered], y_all[covered])
        final = sinkhorn(probs, s) if s else probs
        pd.DataFrame({"path": test_df["path"],
                      "label": [classes[i] for i in final.argmax(1)]}).to_csv(top / "submission.csv", index=False)
        cnt = np.bincount(final.argmax(1), minlength=len(classes))
        print(f"  predicted per-class: min={cnt.min()} median={int(np.median(cnt))} max={cnt.max()}"
              f"  (balanced would be ~{len(test_df) / len(classes):.1f})")
        print(f"  -> wrote {top / 'submission.csv'} from {len(tests)} backbone(s)")

    if not tests:
        raise RuntimeError("No backbone finished; increase TIME_BUDGET_MIN.")
    return top / "submission.csv"


if __name__ == "__main__":
    run(Config())
