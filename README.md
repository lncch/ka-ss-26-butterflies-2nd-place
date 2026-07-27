# 2nd Place Solution (Private LB: 0.9075)

Thanks to KAUST Academy for hosting this one. The dataset had a trick hidden in it that made this a
genuinely fun competition to debug, and once you find it the score jumps from 0.54 to 0.90+.

## Summary

The whole competition comes down to one thing:

**The test set is 100% grayscale. The training set is 100% colour.**

Colour is a huge part of how you tell these species apart, so a colour-trained model falls apart at
test time. That's exactly what the baseline notebook shows: CV 0.92 but LB 0.54. It isn't
overfitting, it's a domain shift.

Convert **every** image to grayscale (train, validation and test) and the gap closes. Everything
after that is just normal careful image classification: a few diverse backbones, moderate
augmentation, flip TTA, and an equal-weight ensemble.

## How I found it

I compared the two sets property by property before touching a model:

| check | train | test |
|---|---|---|
| images | 6000 (100 classes × exactly 60) | 1480 |
| dimensions | all 224×224 RGB JPEG | all 224×224 RGB JPEG |
| JPEG quantization tables | luminance 369, chroma 558 (≈ q95) | **byte-identical** |
| **exactly grayscale (`R==G==B` every pixel)** | **0 / 6000** | **1480 / 1480** |

Everything matches except the last row, which is why nothing looks wrong at a glance. The check
itself is four lines:

```python
a = np.asarray(Image.open(path).convert('RGB'))
is_gray = np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2])
```

What tipped me off first was file size: test images are ~14% smaller than train images at identical
dimensions (median 22.0 KB vs 25.5 KB). Same pixel count, less data, so something had been stripped out.

## Confirming it before burning GPU time

I didn't want to commit to a rewrite on a hunch, so I ran a frozen-ResNet50 linear probe first.
Extract features, fit logistic regression, then score three combinations on the same held-out images:

| condition | macro-F1 |
|---|---|
| colour-train → gray-val | **0.566** |
| colour-train → colour-val | **0.868** |
| **gray-train → gray-val** | **0.841** |

The first row reproduces the baseline's 0.54 LB and the second reproduces its 0.874 CV, both within
0.03, from a linear probe with no fine-tuning at all. That was enough to be confident the diagnosis
was right and the third row was the fix.

## Which grayscale conversion?

Conversion isn't unique, and picking the wrong one leaves a residual gap. Since both sets come from
the same image pool, the right conversion should make the train histogram match the test histogram
best. Wasserstein distance between aggregate grayscale histograms:

| conversion | distance ↓ |
|---|---|
| **ITU-R 601-2** (`PIL .convert('L')`) | **0.664** |
| ITU-R 709 | 1.526 |
| gamma-correct linear luminance | 3.452 |
| flat channel average | 6.242 |

PIL's default wins by a wide margin. I also re-encode the converted training images at JPEG q95 as
cheap insurance against a compression mismatch.

## Three things to fix in the baseline notebook

- **It builds the model once *outside* the fold loop**, so folds 2+ validate on data the model
  already trained on. You can see it in its own logs: fold 1 climbs to 0.874 over seven epochs, then
  fold 2 *starts* at 0.897 and fold 3 at 0.930. That's what inflates its OOF to 0.926.
- **It selects checkpoints on validation loss.** The metric is macro-F1, so select on that.
- **Validation has to be grayscale too.** If you convert train and test but leave validation in
  colour, your CV goes right back to lying to you.

## Training setup

```
==================== Preprocessing ======================
Image.convert('L').convert('RGB')     # ITU-R 601-2
JPEG re-encode at q95
LUMA_JITTER_PROB   = 0.5              # RGB->gray weights jittered +-0.05 around (.299,.587,.114)

==================== Augmentations ======================
RANDOM_RESIZED_CROP = scale (0.65, 1.0), ratio (0.8, 1.25)
HFLIP               = 0.5
ROTATION_DEGREES    = 15
BRIGHTNESS/CONTRAST = 0.2
ERASING_PROB        = 0.25
MIXUP_ALPHA         = 0.2
CUTMIX_ALPHA        = 1.0
MIX_PROB            = 0.5      # switch_prob 0.5 between mixup and cutmix
LABEL_SMOOTHING     = 0.1
NO hue/saturation jitter (meaningless once the image is grayscale)

==================== Training ===========================
OPTIMIZER = AdamW, lr 3e-4, wd 0.05
SCHEDULE  = cosine + warmup, AMP fp16
SPLIT     = single stratified 80/20 per model (not k-fold)
EPOCHS    = 12-18
TTA       = hflip
=========================================================
```

Augmentation is deliberately moderate. With only 224×224 source images and 60 per class, heavy
distortion hurt more than it helped in my runs.

On the split: I used one 80/20 split per model instead of k-fold. Each model then trains on 80% of
the data, and the same GPU budget buys **diverse architectures** instead of repeated folds of one
backbone. That trade paid off more than anything else I tried.

## Final ensemble

Four models, equal-weight probability average:

| model | epochs | seed | inference |
|---|---|---|---|
| `convnext_base.fb_in22k_ft_in1k` | 18 | 61 | 256px |
| `convnext_base.fb_in22k_ft_in1k` | 12 | 11 | 256px |
| `deit3_base_patch16_224.fb_in22k_ft_in1k` | 12 | 51 | 224px |
| `swin_small_patch4_window7_224.ms_in22k_ft_in1k` | 12 | 7 | 224px |

Individually these scored 0.938-0.949 on their holdouts. Mixing families is what matters more than
the individual numbers. ConvNeXt and the transformers agree on only ~95% of test predictions, and
the average feeds on that disagreement. Equal weights beat every weighting scheme I tried.

**The two ConvNeXts are evaluated at 256px despite training at 224.** `RandomResizedCrop` magnifies
objects during training relative to a plain resize at test time, so testing a bit larger can restore
the scale match (the FixRes effect). Being honest about this one: on my holdouts it gained +0.0033 on
one ConvNeXt and *lost* 0.0026 on the other, so it's not a reliable free win. Validate it yourself
rather than assuming. ConvNeXt is fully convolutional so it tolerates the resolution change; DeiT3
and Swin have a fixed patch grid and have to stay at 224.

## Running it

Everything is in one notebook, [`notebooks/train_gray.ipynb`](notebooks/train_gray.ipynb). Import it
to Kaggle, add the competition data, turn Internet on for the `timm` weights, and Run All. About an
hour on a single GPU. It trains all four members, blends them and writes `submission.csv`, rewriting
the submission after each member so an interrupted run still gives you something valid.

Thanks for reading!
