# KA_SS_26 Challenge 1 — competition rules and hard constraints

**Read this before writing or changing any code in this repo.** These are the organizers' rules plus the
facts about this dataset that any implementation must respect. Violating the first section can
disqualify the submission.

## Rules (from the organizers' slides — non-negotiable)

1. **No external data.** All training must use only the provided dataset. The 1480 test images count as
   part of the provided dataset, but nothing from outside it — no scraped images, no other butterfly
   datasets, no synthetic data generated from an outside model.
2. **Pretrained weights: generic ImageNet only.** timm tags resolving to `in1k` / `in12k` / `in21k` /
   `in22k` are fine (ImageNet-21k is still ImageNet).
   **Forbidden:** CLIP/LAION/OpenAI, DINOv2, EVA, JFT, WebLI, MERGED-2B, DFN, SigLIP — and above all
   **iNaturalist**, which contains butterfly species and would be domain-specific external data.
   Enforced in code by `assert_imagenet_only()`, which inspects the tag timm actually resolves
   (`get_pretrained_cfg`) rather than the model string, because tags carry vendor prefixes
   (`fb_in22k_ft_in1k`) and bare arch names resolve to tags the name never spells out.
3. **AI tools are allowed**, provided you understand what you are doing.
4. **TAs may answer questions but may not write code** for participants.
5. **Compute: Colab free tier / Kaggle only.** GPUs allowed: T4, 2xT4, P100.
   Note: Kaggle's current PyTorch build is sm_70+, so **P100 (sm_60) fails on every CUDA kernel**
   with `cudaErrorNoKernelImageForDevice`. Use T4.
6. **Top 3 teams must share their solution** after the competition closes.

## Metric

**Macro-averaged F1** — `f1_score(y_true, y_pred, average='macro')`, confirmed on the competition page.
Every class counts equally regardless of frequency. Select checkpoints on macro-F1, never on loss.

## The organizers' four hints (they were memes, and all four were real)

1. **Do EDA.** — The decisive finding was only visible by looking at the pixels.
2. **The public LB is not reliable.** — 1480 test images over 100 classes; the public slice is a
   fraction of that, so only a handful of images per class decide each class's F1. Choose submissions on
   OOF, never on public-LB movements.
3. **Do not paste in a giant augmentation stack.** — Keep augmentation moderate and justified.
4. **"Why is my public LB so bad?" → "THE DATA."** — See below.

## Dataset facts (measured, not assumed)

| | train | test |
|---|---|---|
| images | 6000 (100 classes x exactly 60) | 1480 |
| **exactly grayscale (R==G==B every pixel)** | **0 / 6000** | **1480 / 1480** |
| dimensions | all 224x224 RGB-mode JPEG | all 224x224 RGB-mode JPEG |
| JPEG quantization table | sum 369 (~PIL quality 95) | byte-identical |
| exact duplicates | none | none |
| train/test overlap | none (min pHash Hamming 4, median 16) | |
| ordering leak | none — adjacent-index embedding similarity is -0.00 sigma vs random | |

**The test set is grayscale and the training set is colour.** Colour is the main cue for species, so a
colour-trained model collapses at test time. This is the whole competition.

**Therefore: every image the model sees — train, validation and test — must be grayscale.**
Validation especially: if val stays in colour, CV lies and you are flying blind again.

The conversion is PIL's ITU-R 601-2 luma (`.convert('L')`), verified by matching aggregate grayscale
histograms against the real test set (Wasserstein 0.664 vs 1.53 for ITU-R 709 and 6.24 for a flat
channel average — an 8x spread, so the test discriminates).

## Reference results

- Organizers' baseline: CV 0.92 / **LB 0.54** (trained on colour, tested on grayscale).
  Its CV was further inflated by building the model *once outside* the fold loop, so folds 2+ validated
  on data they had already trained on.
- Frozen-resnet50 linear probe: colour→gray 0.566, colour→colour 0.868, **gray→gray 0.841**.
- This pipeline: **OOF macro-F1 0.94143** (3 folds, convnext_small), **LB 0.8961**.

## Implementation invariants — do not regress these

- Grayscale applied to train, validation **and** test.
- A fresh model, optimizer, scheduler and scaler for **every** fold.
- Checkpoint selection on validation **macro-F1**, not loss.
- OOF must use the **same TTA** as the test predictions, or CV stops predicting the LB.
- Any prior/threshold correction must be tuned on OOF and applied only if it actually improves it.
- A valid `submission.csv` must exist on disk after every completed fold — sessions die.
- No `torch.cuda.amp` (deprecated); use `torch.amp` with an explicit device.

## Verification before shipping a change

`python3 scripts/smoke_test.py` — runs the real data on CPU in ~1 min and asserts a valid 1480-row
submission, correct row order, valid class names, three identical channels, and a fresh model per fold.
After editing `src/pipeline.py`, run `python3 scripts/sync_notebook.py` to regenerate the notebook;
it checks that every cell parses **on its own**.
