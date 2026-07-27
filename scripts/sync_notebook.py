"""Generate notebooks/train_gray.ipynb from src/pipeline.py.

The notebook is what actually runs on Kaggle, so it must never drift from the reviewed
source. src/pipeline.py is split on its `# %% <title>` markers, one notebook cell per
section, each preceded by a markdown heading. Re-run this after editing pipeline.py.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks/train_gray.ipynb"
SRC = ROOT / "src/pipeline.py"
MAIN = 'if __name__ == "__main__":\n    run(Config())\n'

INTRO = """# KA_SS_26 Challenge 1 — grayscale-matched classifier

**The test set is 100% grayscale; the training set is 100% colour.** Measured on the real data:
1480/1480 test images have `R == G == B` on every pixel, 0/6000 train images do. Both sets are
224x224 JPEG with identical quantization tables (~quality 95). Colour is the main cue for species,
so a colour-trained model collapses at test time — that is the published baseline's CV 0.92 / LB 0.54.

A frozen-resnet50 linear probe reproduces both numbers and the fix:

| condition | macro-F1 | corresponds to |
|---|---|---|
| colour-train -> gray-val | 0.566 | the 0.54 LB |
| colour-train -> colour-val | 0.868 | the 0.874 clean fold-1 CV |
| **gray-train -> gray-val** | **0.841** | **the fix** |

Every image here — train, validation and test — is converted to grayscale, so CV finally measures
what the leaderboard measures. Select a GPU accelerator, turn Internet on (for the timm weights),
and run top to bottom.
"""

CONFIG = """# CONFIG — all user-facing knobs live in this one cell.
# Measured on real T4 runs: ONE fold of a big backbone beats THREE folds of a small one, and
# costs a third as much. 1 fold trains on 80% of the data (3 folds train on only 67%).
#   convnext_base  1 fold, 12 epochs, 13 min  ->  OOF 0.94462
#   swin_small     1 fold, 12 epochs, 13 min  ->  OOF 0.94198
#   convnext_small 3 folds, 18 epochs, ~45min ->  OOF 0.94143
# So: spend the budget on diverse strong backbones, not on more folds of one.
ENSEMBLE_MODELS = [
    'convnext_base.fb_in22k_ft_in1k',                 # best single model measured
    'swin_base_patch4_window7_224.ms_in22k_ft_in1k',  # transformer — decorrelates the convnets
    'tf_efficientnetv2_m.in21k_ft_in1k',              # third family
    'deit3_base_patch16_224.fb_in22k_ft_in1k',        # second transformer; drop if short on time
]
IMG_SIZE = 224
EPOCHS = 12
FOLDS = 1                         # 1 fold = trains on 80%; more backbones beat more folds
RUN_MODE = 'full'
TIME_BUDGET_MIN = 75              # split across ALL backbones; set to real remaining time
USE_EMA = False
P_LUMA_JITTER = 0.5
MATCH_JPEG = True
VERTICAL_FLIP = False
BATCH_SIZE = 64
NUM_WORKERS = 4
DATA_ROOT = '/kaggle/input/competitions/ka-ss-26-challenge-1'   # auto-detected if this misses
OUTPUT_DIR = '/kaggle/working'
"""

TAIL_CELLS = [
    ("md", "## Locate the data and verify the domain\n\n"
           "Works whether the competition is attached as a notebook Input or fetched with "
           "`kagglehub.competition_download`. The sanity check fails loudly if the test set is "
           "ever swapped for a colour one."),
    ("code", "DATA_ROOT = str(resolve_data_root(DATA_ROOT))\n"
             "print('Data root:', DATA_ROOT)\n"
             "for _m in ENSEMBLE_MODELS:\n"
             "    print(f'  weights OK: {_m}  (tag={assert_imagenet_only(_m)}, ImageNet-only)')\n"
             "sanity_check_grayscale(DATA_ROOT, samples=None)   # asserts every test image is R == G == B\n"),
    ("md", "## Train, validate, predict\n\n"
           "Each backbone trains into its own subfolder, then their test probabilities are averaged.\n"
           "`submission.csv` is rewritten after every fold *and* after every backbone, and the OOF array\n"
           "is saved as it goes — so you can stop at any point and still have a valid ensemble.\n\n"
           "Watch the `OOF ENSEMBLE` line: that is the measured score of the blend, not a guess."),
    ("code", "cfg = Config(\n"
             "    DATA_ROOT=DATA_ROOT, OUTPUT_DIR=OUTPUT_DIR,\n"
             "    IMG_SIZE=IMG_SIZE, EPOCHS=EPOCHS, FOLDS=FOLDS,\n"
             "    RUN_MODE=RUN_MODE, TIME_BUDGET_MIN=TIME_BUDGET_MIN, USE_EMA=USE_EMA,\n"
             "    P_LUMA_JITTER=P_LUMA_JITTER, MATCH_JPEG=MATCH_JPEG,\n"
             "    VERTICAL_FLIP=VERTICAL_FLIP, BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS,\n"
             ")\n"
             "submission_path = run_ensemble(cfg, ENSEMBLE_MODELS)\n"
             "print('Submission ready:', submission_path)\n"),
    ("md", "## Check the submission"),
    ("code", "import pandas as pd\n"
             "sub = pd.read_csv(submission_path)\n"
             "print(sub.shape)\n"
             "print('distinct classes predicted:', sub['label'].nunique(), 'of 100')\n"
             "print('most / least predicted:', sub['label'].value_counts().iloc[0],\n"
             "      '/', sub['label'].value_counts().iloc[-1])\n"
             "sub.head()\n"),
]


def cell(kind: str, text: str) -> dict:
    src = text.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": src}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def main() -> None:
    code = SRC.read_text()
    assert code.endswith(MAIN), "src/pipeline.py no longer ends with the expected __main__ block"
    body = code[: -len(MAIN)].rstrip() + "\n"

    cells = [cell("md", INTRO), cell("code", CONFIG)]
    chunks = body.split("# %% ")
    assert chunks[0].strip().startswith('"""'), "expected the module docstring before the first marker"

    sections = []
    for chunk in chunks[1:]:
        title, _, rest = chunk.partition("\n")
        sections.append([title.strip(), rest.strip().splitlines()])

    # A marker placed just under a decorator would strand it at the end of a cell, where it
    # is a SyntaxError on its own. Carry any trailing decorator lines to the next section.
    for cur, nxt in zip(sections, sections[1:]):
        moved = []
        while cur[1] and cur[1][-1].lstrip().startswith("@"):
            moved.insert(0, cur[1].pop())
        if moved:
            nxt[1][:0] = moved
            print(f"  moved trailing decorator(s) {moved} into section {nxt[0]!r}")

    for title, lines in sections:
        cells += [cell("md", f"### {title}"), cell("code", "\n".join(lines).strip() + "\n")]
    cells += [cell(k, t) for k, t in TAIL_CELLS]

    # Every cell must be valid Python *on its own* — joining them first hides dangling
    # decorators and other split-point damage.
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        text = "".join(c["source"])
        try:
            ast.parse(text)
        except SyntaxError as exc:
            raise SystemExit(f"cell {i} is not valid Python on its own: {exc}\n---\n{text}")

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"Wrote {NB.relative_to(ROOT)}: {len(cells)} cells ({n_code} code, {len(cells)-n_code} markdown)")


if __name__ == "__main__":
    main()
