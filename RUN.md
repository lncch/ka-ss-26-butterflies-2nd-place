# How to run this on Kaggle

## The one-line version

The test set is 100% grayscale, the train set is 100% colour. Train on grayscale and the 0.30 gap closes.
Everything else in here is a normal, careful image-classification pipeline.

## Evidence (measured on the real data, not assumed)

| check | train | test |
|---|---|---|
| images | 6000 (100 classes x exactly 60) | 1480 |
| exactly grayscale (R==G==B every pixel) | **0 / 6000** | **1480 / 1480** |
| dimensions | all 224x224 RGB JPEG | all 224x224 RGB JPEG |
| JPEG quantization table | sum 369 (~quality 95) | identical |
| exact duplicates / train-test overlap | none | none |

Frozen-resnet50 linear probe, same held-out split, macro-F1:

| condition | score | what it corresponds to |
|---|---|---|
| train colour -> validate gray | 0.566 | the starter notebook's real LB (0.54) |
| train colour -> validate colour | 0.868 | the starter's own reported CV (0.874) |
| **train gray -> validate gray** | **0.841** | **the fix** |

## Setup

1. kaggle.com -> Code -> New Notebook -> File -> Import Notebook -> upload `notebooks/train_gray.ipynb`.
2. Right panel: **Input** -> Add Input -> the competition dataset.
   Confirm the path is `/kaggle/input/competitions/ka-ss-26-challenge-1` (matches `DATA_ROOT`).
3. Settings -> **Accelerator: GPU T4 x2** (or a single T4), **Internet: ON**.
   Do **not** pick P100: Kaggle's PyTorch build supports sm_70+ and the P100 is sm_60, so every
   CUDA kernel fails with `cudaErrorNoKernelImageForDevice`. T4 is sm_75 and works.
   Internet is required once, so `timm` can fetch the ImageNet weights.
4. Run all.

## Run order — bank a score first, then improve

**Step 1 (~25 min): `RUN_MODE='quick'`.** One fold, 8 epochs. Submit it immediately.
This guarantees a real score on the board and confirms the whole path works end-to-end before you
spend hours on it. Expect a large jump over the 0.54 baseline.

**Step 2 (~2-3 h): `RUN_MODE='full'`.** 5 folds, 18 epochs, `TIME_BUDGET_MIN` set to the time you
actually have. A submission is rewritten after every fold, so you can stop at any point and still submit.

**Step 3 (if time remains): a second backbone.** Change `MODEL_NAME` to
`tf_efficientnetv2_s.in21k_ft_in1k`, run again, then average the saved `test_probs.npy` files from the two
runs. Different architecture families make the most useful ensemble partners.

## Config knobs that matter

| knob | default | note |
|---|---|---|
| `RUN_MODE` | `full` | `smoke` (local CPU test) / `quick` (bank a score) / `full` |
| `TIME_BUDGET_MIN` | 180 | set to the real time you have; new folds stop launching before it is blown |
| `MODEL_NAME` | `convnext_small.fb_in22k_ft_in1k` | see `MODEL_ZOO` for rule-compliant alternates |
| `IMG_SIZE` | 224 | native size. 288 is worth one try; do not bother with 384 |
| `P_LUMA_JITTER` | 0.5 | randomises the RGB->gray weights slightly. Set 0.0 to disable |
| `MATCH_JPEG` | True | adds the second JPEG generation the test images have |

## Rule compliance

- No external data; only the provided dataset is read.
- `assert_imagenet_only` resolves the tag timm will actually download (`get_pretrained_cfg`) and
  requires it to name an ImageNet corpus (`in1k`/`in12k`/`in21k`/`in22k`). ImageNet-21k is still
  ImageNet. It matches on the resolved tag rather than the model string, because tags carry a vendor
  prefix (`fb_in22k_ft_in1k`) and a bare arch name resolves to a default tag the name never spells out.
- Explicitly **not** used: CLIP/LAION, DINOv2, EVA, JFT, iNaturalist, or any other non-ImageNet corpus.
  iNaturalist in particular would be disqualifying here — it contains butterfly species.
- Trains and infers inside a single Kaggle GPU session.

## Choosing what to submit

The hints tell you the public LB is unreliable, and they are right: the public split is a slice of only
1480 images across 100 classes, so a couple of images per class decide each class's F1.

Pick your final submissions on **OOF macro-F1**, not on the public LB. If a change gains on the public LB
but loses on OOF, it is noise — do not take it. Submit your best OOF ensemble, and use the second slot for
a more conservative variant (e.g. fold-ensemble without the prior correction) as a hedge.

## Optional, only with time to spare

- **Pseudo-labelling the test set.** The 1480 test images are part of the provided dataset, so using them
  unlabelled is within the letter of the rules — but confirm with a TA before relying on it. Take
  high-confidence test predictions, add them to training, retrain one fold. This adapts the model to the
  test distribution and is the largest remaining lever.
- **Label cleaning.** Look for training images the OOF ensemble confidently disagrees with, and inspect
  them by eye. The dataset looked clean on duplicates, but visual mislabels are not detectable by hashing.
