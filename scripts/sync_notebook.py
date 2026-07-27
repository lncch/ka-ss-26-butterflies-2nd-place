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

CONFIG = """# CONFIG
# The four ensemble members, exactly as submitted. Each trains on its own stratified 80/20 split.
# ConvNeXt is fully convolutional so it can be evaluated above its training resolution (FixRes);
# DeiT3 and Swin have a fixed patch grid and must stay at 224.
MEMBERS = [
    {'model': 'convnext_base.fb_in22k_ft_in1k',                  'epochs': 18, 'seed': 61, 'infer_size': 256},
    {'model': 'convnext_base.fb_in22k_ft_in1k',                  'epochs': 12, 'seed': 11, 'infer_size': 256},
    {'model': 'deit3_base_patch16_224.fb_in22k_ft_in1k',         'epochs': 12, 'seed': 51, 'infer_size': 224},
    {'model': 'swin_small_patch4_window7_224.ms_in22k_ft_in1k',  'epochs': 12, 'seed':  7, 'infer_size': 224},
]
IMG_SIZE = 224                    # training resolution; the source images are natively 224
TIME_BUDGET_MIN = 240             # stops starting new members once spent
BATCH_SIZE = 64
NUM_WORKERS = 4
P_LUMA_JITTER = 0.5               # randomises the RGB->gray weights slightly during training
MATCH_JPEG = True                 # re-encode converted training images at q95
DATA_ROOT = '/kaggle/input/competitions/ka-ss-26-challenge-1'   # auto-detected if this misses
OUTPUT_DIR = '/kaggle/working'
"""

TAIL_CELLS = [
    ("md", "## Locate the data and verify the domain\n\n"
           "The assertion below is the whole premise of this solution: every test image must have "
           "identical R, G and B channels. If that ever stops being true, nothing downstream is valid."),
    ("code", "DATA_ROOT = str(resolve_data_root(DATA_ROOT))\n"
             "print('Data root:', DATA_ROOT)\n"
             "for _m in MEMBERS:\n"
             "    print(f\"  weights OK: {_m['model']}  (tag={assert_imagenet_only(_m['model'])}, ImageNet-only)\")\n"
             "sanity_check_grayscale(DATA_ROOT, samples=None)   # asserts all 1480 test images are R == G == B\n"),
    ("md", "## Train all four members and blend\n\n"
           "Each member trains on its own 80/20 split, predicts with hflip TTA at its own inference\n"
           "resolution, and the four are averaged with equal weight. `submission.csv` is rewritten after\n"
           "every member, so stopping early still leaves a valid submission.\n\n"
           "On a single T4 the whole run takes roughly an hour."),
    ("code", "cfg = Config(\n"
             "    DATA_ROOT=DATA_ROOT, OUTPUT_DIR=OUTPUT_DIR, IMG_SIZE=IMG_SIZE,\n"
             "    TIME_BUDGET_MIN=TIME_BUDGET_MIN, BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS,\n"
             "    P_LUMA_JITTER=P_LUMA_JITTER, MATCH_JPEG=MATCH_JPEG,\n"
             ")\n"
             "submission_path = run_solution(cfg, MEMBERS)\n"
             "print('Submission ready:', submission_path)\n"),
    ("md", "## Check the submission"),
    ("code", "import pandas as pd\n"
             "sub = pd.read_csv(submission_path)\n"
             "print(sub.shape, '| distinct classes predicted:', sub['label'].nunique(), 'of 100')\n"
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
