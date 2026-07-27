# KA_SS_26 Challenge 1 — Solution Write-up

**Task:** 100-species butterfly/moth classification, macro-F1.

**Result: 2nd place** — public **0.9183** → private **0.9075**, up from a 0.54 baseline.
(The top entry on the final leaderboard is the organizers' baseline and does not count.)

The scoring submission was a 4-model probability average: `convnext_base@18ep` and
`convnext_base@12ep` (both evaluated at 256px) + `deit3_base` + `swin_small`, all grayscale-matched
with hflip TTA. It was our top public score *and* the second-best private score of our 27
submissions.

---

## TL;DR

**The test set is 100% grayscale. The training set is 100% colour.**

Colour carries a large share of the discriminative signal for these species, so a colour-trained
model degrades badly at test time: in our probe, training on colour and testing on grayscale costs
0.30 macro-F1 versus testing on colour.
That single fact is the core of the competition — it accounts for the published baseline's
CV 0.92 / LB 0.54 gap. Convert *every* image (train, validation and test) to grayscale and the gap closes.

Everything after that is ordinary careful image classification.

---

## 1. Finding it

The organizers' hints said "do EDA" and "why is my public LB so bad? → THE DATA". So before touching
a model, we compared the two sets pixel by pixel:

| check | train | test |
|---|---|---|
| images | 6000 (100 classes × exactly 60) | 1480 |
| dimensions | all 224×224 RGB-mode JPEG | all 224×224 RGB-mode JPEG |
| JPEG quantization tables | luminance sum 369, chroma 558 (≈ PIL quality 95) | **byte-identical** |
| **exactly grayscale (`R==G==B` every pixel)** | **0 / 6000** | **1480 / 1480** |

The first three rows match perfectly, which is why nothing looked wrong at a glance. The fourth is
decisive: every test image has identical R, G and B channels; not one training image does.

The tell that led there was cheap — test files were ~14% smaller than train files at identical
dimensions (median 22.0 KB vs 25.5 KB). Same size, less data, so something had been removed.

## 2. Confirming it before spending any GPU time

A frozen-ResNet50 linear probe on our own held-out split lands close to the competition's published
numbers. It is a different model on different labelled images, so it corroborates the diagnosis
rather than reproducing the leaderboard:

| condition | macro-F1 | corresponds to |
|---|---|---|
| colour-train → gray-val | **0.566** | the published **LB 0.54** |
| colour-train → colour-val | **0.868** | the published **CV 0.874** (clean fold) |
| **gray-train → gray-val** | **0.841** | **the fix** |

Both published numbers are matched to within ~0.03 by a linear probe with no fine-tuning. That was
enough to commit GPU time to the fix.

## 3. Picking the closest conversion

Grayscale conversion is not unique, and a mismatch would leave a residual train/test gap. Assuming
both sets are drawn from the same image pool, the conversion we apply to train should bring its
aggregate grayscale histogram closest to the test set's. This ranks candidates; it does not prove
which function the organizers called. Wasserstein distance, train-converted-under-X vs real test:

| conversion | distance ↓ |
|---|---|
| **ITU-R 601-2** (`PIL .convert('L')`, OpenCV) | **0.664** |
| ITU-R 709 | 1.526 |
| gamma-correct linear luminance | 3.452 |
| flat channel average | 6.242 |

The 5.58 spread is 8× the best distance, so the comparison discriminates between candidates, and
PIL's default is the clear winner among those tested. We also re-encode the converted training
images at JPEG q95. Note this is a guess: identical quantization tables do not reveal how many times
an image was compressed, so we cannot show the test images carry a second generation — the
re-encode is cheap insurance, not a verified match.

## 4. Fixing the validation

The starter notebook creates its model **once, outside the fold loop**, so folds 2+ validate on data
the model already trained on. The symptom is visible in its own logs:

```
fold 1: val_f1 rises to 0.874 over 7 epochs
fold 2: val_f1 STARTS at 0.897     <- model already saw this data
fold 3: val_f1 STARTS at 0.930
```

That inflated its reported OOF from ~0.874 to 0.926. Fixes applied:

- fresh model, optimizer, scheduler and AMP scaler **every fold**
- checkpoint selection on validation **macro-F1**, not validation loss
- the **final OOF predictions use the same TTA as the test predictions** (hflip), so the number we
  select on measures what we actually submit. Per-epoch validation during training runs without TTA
  — it only has to rank checkpoints, and skipping it there saves a full pass per epoch

## 5. Final recipe

- **Preprocessing:** `Image.convert('L').convert('RGB')` + JPEG q95 re-encode, pre-decoded once into
  a shared `uint8` array (fork-shared across dataloader workers).
- **Augmentation, deliberately moderate:** RandomResizedCrop(0.65–1.0), hflip, ±15° rotation,
  brightness/contrast 0.2, RandomErasing 0.25, mixup 0.2 / cutmix 1.0 @ p=0.5, label smoothing 0.1.
  **No hue/saturation jitter** — meaningless once the image is grayscale.
- **Luma jitter:** with p=0.5, convert RGB→gray using weights jittered ±0.05 around
  (0.299, 0.587, 0.114), hedging against uncertainty in the exact conversion.
- **Backbones** (ImageNet-only weights, per the rules): `convnext_base`, `deit3_base`, `swin_small`,
  `swin_base`, `convnext_small`, `tf_efficientnetv2_s/m`.
- **Training:** AdamW lr 3e-4, wd 0.05, cosine + warmup, AMP fp16, 12–18 epochs.
- **Inference:** hflip TTA, probability averaging. The scoring submission averaged **4 models drawn
  from 3 architecture families** (two ConvNeXt-Base runs, DeiT3-Base, Swin-Small).

### Compute note

We ran models **headlessly via the Kaggle API** (`kaggle kernels push`) rather than interactively,
two concurrent T4 kernels at a time. That trained 8 models in the time one interactive 3-fold run
took. Two gotchas: the accelerator string is case-sensitive (`NvidiaTeslaT4`; a wrong value silently
falls back to **P100, which is currently broken on Kaggle** — sm_60 against an sm_70+ torch build),
and a single T4 beats T4×2 unless your code is actually multi-GPU.

## 6. What we measured that did NOT work

Negative results, so nobody re-runs them:

- **No leakage found by the probes we ran.** MD5 and pHash found no exact duplicates and no
  train/test hash collisions; nearest-neighbour embedding cosine from test to train peaks at 0.978
  (nothing near-identical); and adjacent test indices are no more similar in embedding space than
  random pairs (−0.00σ), so we found no index-order structure. These rule out the specific forms of
  leakage we tested, not all conceivable ones.
- **Balanced assignment (Sinkhorn).** Train is exactly 60/class, so the test set is plausibly
  balanced too, which under macro-F1 would make constraining the predicted marginal toward uniform
  worthwhile. (We could not verify this — test labels are hidden, and the fact that 1480 decomposes
  as 80×15 + 20×14 is just arithmetic, not evidence.) Measured on 6000 OOF rows it gained
  **+0.00047** — indistinguishable from noise. Dropped.
- **FixRes (testing at higher resolution than training).** Gained +0.0033 on one convnext, *lost*
  0.0026 on another. Inconsistent, so not a real effect — though the submission built on it did
  score well, which is itself a lesson about how noisy the feedback was.
- **Bigger ensembles with quality weighting.** Blending 8 models scored *worse* on public than 3.
  See the caveat below — we no longer believe the public LB could resolve this.

## 7. How much can you trust the public LB here? (and how we chose finals)

Across our 27 scored submissions, **public and private correlated only +0.38**:

```
                                 public   private
scoring submission (z1)          0.9183   0.9075    top public, 2nd-best private
best private available (g2)      0.9074   0.9205    public rank #10 of 27
the 8-model blend we dismissed   0.8952   0.9053    worst public, 7th-best private
one we selected                  0.9178   0.8971    2nd public, 21st private
```

Changing **9 of 1480 predictions moved the public score by 1.5 points** — roughly 3× what that many
label flips should cost under a uniform model. Macro-F1 is nonlinear, so this is evidence that the
public score is *unstable* rather than a measurement of how small the subset is; either way, gaps of
1–2 points between similar submissions carry little information.

We got lucky: our top-public pick also happened to be our 2nd-best private. But the single best
private submission sat at public rank #10, and the blend with the *worst* public score of all
finished 7th-best private. Ranking on public would not reliably have found either.

**The rule worth taking away:** with N final slots and a noisy public score, the slots are N
*independent bets*, not N guesses at the same answer. Spend one on your best-measured candidate and
the rest on structurally different ones — different ensemble size, different architecture mix — so
that a single wrong hypothesis cannot sink all N. Our three selections were near-identical variants
of one idea, so they would have risen or fallen together; that they rose was not something we had
earned.

A related trap in the other direction: **do not read a low public score as evidence a model is bad**
when you have already established the public set cannot resolve that difference. We dropped the
8-model blend for exactly that reason, and it outscored two of our three finals on private.

## 8. Reproducing

**The scoring submission is not a single notebook run**, and it is worth being precise about that.
It averaged four model outputs from four separate Kaggle kernel runs, two of which were then
re-inferred locally at 256px:

| member | run | inference |
|---|---|---|
| `convnext_base.fb_in22k_ft_in1k` | seed 61, 18 epochs | **256px** (FixRes) |
| `convnext_base.fb_in22k_ft_in1k` | seed 11, 12 epochs | **256px** (FixRes) |
| `deit3_base_patch16_224` | seed 51, 12 epochs | 224px |
| `swin_small_patch4_window7_224` | seed 7, 12 epochs | 224px |

All four use `FOLDS=1` (each trains on 80% of the data), grayscale matching and hflip TTA, and are
averaged with equal weight.

```
README.md                       this write-up
RULES.md                        competition rules, dataset facts, implementation invariants
RUN.md                          how to run it on Kaggle

notebooks/train_gray.ipynb      self-contained Kaggle notebook (generated from src/pipeline.py)
src/pipeline.py                 the training/inference pipeline
scripts/sync_notebook.py        regenerates the notebook from pipeline.py (keeps the two in sync)
scripts/push_kernel.py          patches the config and pushes one backbone as a headless kernel
scripts/fixres.py               256px re-inference + its holdout validation
scripts/fixres_single.py        the same for one checkpoint
scripts/reproduce_best.py       averages the four members into the exact scoring submission
scripts/smoke_test.py           real-data CPU end-to-end test (~1 min)
scripts/probe_grayscale.py      the linear probe behind §2
scripts/identify_conversion.py  the histogram matching behind §3
scripts/probe_test_structure.py the leakage / ordering probes behind §6
artifacts/probs/*.npy           the two 256px probability arrays the scoring blend needs
submissions.csv                 all 27 scored submissions, public and private (§7)
```

The competition data is not redistributed here. Download it with
`kaggle competitions download -c ka-ss-26-challenge-1` and unzip into `data/raw/`.

To train one model end to end, set `RUN_MODE = 'quick'` in the notebook's config cell (or pass
`Config(RUN_MODE='quick')` when using `src/pipeline.py`) for a score in ~25 minutes, then `'full'`.
To rebuild the exact scoring submission from saved outputs, run `scripts/reproduce_best.py`.

A caveat on the FixRes step: it is why this blend beat the plain 224px version on the leaderboard,
but its holdout validation was *inconsistent* — +0.0033 on one convnext, −0.0026 on the other. We
kept it because the submission scored well, not because we could show it works. Anyone rebuilding
this should validate it themselves rather than assume the gain.
