# 2nd Place — KA_SS_26 Challenge 1

100-species butterfly/moth classification, metric **macro-F1**.
**Final score: 0.9075 private / 0.9183 public**, against a published baseline of 0.54.

The complete solution is [`notebooks/train_gray.ipynb`](notebooks/train_gray.ipynb) — import it to
Kaggle, add the competition data, and run it top to bottom.

---

## Summary

**The test set is 100% grayscale. The training set is 100% colour.**

Colour carries a large share of the discriminative signal for these species, so a colour-trained
model degrades badly at test time — training on colour and validating on grayscale costs 0.30
macro-F1 in a controlled probe. This is what produces the baseline's CV 0.92 / LB 0.54 gap.

Convert **every** image — train, validation and test — to grayscale, and the gap closes. Everything
after that is ordinary careful image classification.

---

## 1. The finding

Comparing the two sets property by property:

| check | train | test |
|---|---|---|
| images | 6000 (100 classes × exactly 60) | 1480 |
| dimensions | all 224×224 RGB-mode JPEG | all 224×224 RGB-mode JPEG |
| JPEG quantization tables | luminance sum 369, chroma 558 (≈ quality 95) | **byte-identical** |
| **exactly grayscale (`R==G==B` every pixel)** | **0 / 6000** | **1480 / 1480** |

The first three rows match perfectly, which is why nothing looks wrong at a glance. The fourth is
decisive: every test image has identical R, G and B channels, and not one training image does.

The whole check is four lines:

```python
a = np.asarray(Image.open(path).convert('RGB'))
is_gray = np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2])
```

The cheap tell that led there: test files are ~14% smaller than train files at identical dimensions
(median 22.0 KB vs 25.5 KB). Same pixel count, less data — something had been removed.

## 2. Verifying it before training anything

Before spending any GPU time, a frozen-ResNet50 linear probe: extract penultimate features, fit
logistic regression on a stratified 80/20 split, and score the three combinations on the *same*
held-out images.

| condition | macro-F1 |
|---|---|
| colour-train → gray-val | **0.566** |
| colour-train → colour-val | **0.868** |
| **gray-train → gray-val** | **0.841** |

The first row lands within 0.03 of the published 0.54 leaderboard score and the second within 0.03
of its 0.874 clean-fold CV — strong corroboration of the diagnosis from a probe with no fine-tuning
at all. The third row is the fix, and it is what justified committing to the approach.

## 3. Choosing the conversion

Grayscale conversion is not unique, and a mismatch would leave a residual train/test gap. Assuming
both sets come from the same image pool, the conversion applied to train should bring its aggregate
grayscale histogram closest to the test set's. Converting a size-matched sample of training images
under each candidate and measuring Wasserstein distance to the real test histogram:

| conversion | distance ↓ |
|---|---|
| **ITU-R 601-2** (`PIL .convert('L')`, OpenCV) | **0.664** |
| ITU-R 709 | 1.526 |
| gamma-correct linear luminance | 3.452 |
| flat channel average | 6.242 |

The spread is 8× the best distance, so the comparison discriminates between candidates and PIL's
default is the clear winner among those tested. This ranks candidates; it does not prove which
function was used to build the test set.

Converted training images are also re-encoded at JPEG q95. Identical quantization tables do not
reveal how many times an image was compressed, so this is cheap insurance against a compression
mismatch rather than a verified match.

## 4. Validation that actually predicts the leaderboard

Three things matter, and the published starter gets all three wrong:

- **A fresh model, optimizer, scheduler and AMP scaler every fold.** The starter builds its model
  once *outside* the fold loop, so folds 2+ validate on data the model already trained on. Its own
  logs show it: fold 1 climbs to 0.874 over seven epochs, then fold 2 *starts* at 0.897 and fold 3
  at 0.930. That inflates its reported OOF from ~0.874 to 0.926.
- **Select checkpoints on validation macro-F1**, not validation loss — the metric is macro-F1.
- **Validation images get the same grayscale transform as test**, and the final held-out predictions
  use the same TTA as the test predictions, so the number being selected on measures what is
  actually submitted. (Per-epoch validation runs without TTA; it only has to rank checkpoints.)

## 5. Training recipe

- **Preprocessing:** `Image.convert('L').convert('RGB')` + JPEG q95 re-encode, pre-decoded once into
  a shared `uint8` array.
- **Augmentation, deliberately moderate:** `RandomResizedCrop(scale=0.65–1.0)`, hflip, ±15° rotation,
  brightness/contrast 0.2, `RandomErasing(0.25)`, mixup 0.2 / cutmix 1.0 at p=0.5, label smoothing
  0.1. **No hue/saturation jitter** — meaningless once the image is grayscale.
- **Luma jitter:** with p=0.5, convert RGB→gray using weights jittered ±0.05 around
  (0.299, 0.587, 0.114). Cheap robustness against uncertainty in the exact conversion.
- **Split:** a single stratified 80/20 split per model rather than k-fold, so each model trains on
  80% of the data and the budget buys diverse architectures instead of repeated folds of one.
- **Optimizer:** AdamW, lr 3e-4, wd 0.05, cosine schedule with warmup, AMP fp16, 12–18 epochs.
- **Inference:** hflip TTA, probabilities averaged with equal weight.

## 6. The final ensemble

Four models, equal-weight probability average:

| model | epochs | seed | inference |
|---|---|---|---|
| `convnext_base.fb_in22k_ft_in1k` | 18 | 61 | 256px |
| `convnext_base.fb_in22k_ft_in1k` | 12 | 11 | 256px |
| `deit3_base_patch16_224.fb_in22k_ft_in1k` | 12 | 51 | 224px |
| `swin_small_patch4_window7_224.ms_in22k_ft_in1k` | 12 | 7 | 224px |

Held-out macro-F1 for the individual models ranged 0.938–0.949. Mixing architecture families matters
more than the individual scores: ConvNeXt and the two transformers agree on only ~95% of test
predictions, and it is that disagreement the average exploits.

Two members are evaluated at **256px** despite training at 224. `RandomResizedCrop` magnifies objects
during training relative to a plain resize at test time, so testing at higher resolution can restore
the apparent-scale match (the FixRes effect). Worth knowing before copying it: on our holdouts this
gained +0.0033 on one ConvNeXt and *lost* 0.0026 on the other, so it is not a dependable free win —
validate it on your own data rather than assuming it transfers. ConvNeXt is fully convolutional so it
tolerates the resolution change; DeiT3 and Swin have a fixed patch grid and must stay at 224.

## 7. Running it

Everything is in [`notebooks/train_gray.ipynb`](notebooks/train_gray.ipynb) — it trains all four
members, runs the higher-resolution inference, blends them and writes `submission.csv`.

1. Upload it to Kaggle (**Code → New Notebook → File → Import Notebook**).
2. Add the competition as an Input, set the accelerator to a GPU, and turn Internet on so `timm`
   can fetch the ImageNet weights.
3. Run all. It takes roughly an hour on a single GPU.

`submission.csv` is rewritten after every member finishes, so an interrupted run still leaves a valid
submission — it just blends fewer models. The four members are declared in the first cell, so
changing the ensemble is a one-line edit.
