"""Quantify the color->grayscale domain shift with a frozen-backbone linear probe.

Three conditions, same held-out split:
  A  train COLOR -> validate GRAY   (what the starter notebook effectively does -> LB)
  B  train GRAY  -> validate GRAY   (the proposed fix)
  C  train COLOR -> validate COLOR  (the starter's own CV number, for reference)
"""
import io, os, sys, time
import numpy as np, pandas as pd, torch, timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = os.path.expanduser('~/kaggle/butterflies-c1/data/raw')
DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load(path, gray, match_jpeg=True):
    im = Image.open(path).convert('RGB')
    if gray:
        im = im.convert('L').convert('RGB')
        if match_jpeg:                      # mirror test's extra JPEG generation
            b = io.BytesIO(); im.save(b, 'JPEG', quality=95); im = Image.open(b).convert('RGB')
    return im


class DS(Dataset):
    def __init__(self, paths, root, gray):
        self.paths, self.root, self.gray = paths, root, gray

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        a = np.asarray(load(os.path.join(self.root, self.paths[i]), self.gray), np.float32) / 255.
        return torch.from_numpy(((a - MEAN) / STD).transpose(2, 0, 1))


@torch.no_grad()
def feats(model, paths, root, gray):
    dl = DataLoader(DS(paths, root, gray), batch_size=64, num_workers=4, shuffle=False)
    out = []
    for x in dl:
        out.append(model(x.to(DEV)).float().cpu().numpy())
    return np.concatenate(out)


def main():
    tr = pd.read_csv(f'{ROOT}/train.csv')
    classes = sorted(tr.label.unique())
    y = tr.label.map({c: i for i, c in enumerate(classes)}).values
    itr, iva = train_test_split(np.arange(len(tr)), test_size=0.2, stratify=y, random_state=42)
    ptr, pva = tr.path.values[itr], tr.path.values[iva]
    ytr, yva = y[itr], y[iva]
    print(f'device={DEV}  train={len(itr)}  val={len(iva)}  classes={len(classes)}')

    model = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=0).eval().to(DEV)

    t0 = time.time()
    bank = {}
    for name, paths, gray in [('tr_color', ptr, False), ('tr_gray', ptr, True),
                              ('va_color', pva, False), ('va_gray', pva, True)]:
        bank[name] = feats(model, paths, ROOT, gray)
        print(f'  extracted {name:9s} {bank[name].shape}  ({time.time()-t0:.0f}s)')

    def probe(Xtr, Xva, tag):
        clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
        clf.fit(Xtr, ytr)
        f1 = f1_score(yva, clf.predict(Xva), average='macro')
        print(f'{tag:52s} macro-F1 = {f1:.4f}')
        return f1

    print('\n--- linear probe on frozen resnet50 features ---')
    a = probe(bank['tr_color'], bank['va_gray'], 'A  train COLOR -> val GRAY   (= starter on real test)')
    b = probe(bank['tr_gray'], bank['va_gray'], 'B  train GRAY  -> val GRAY   (= the fix)')
    c = probe(bank['tr_color'], bank['va_color'], 'C  train COLOR -> val COLOR  (= starter\'s own CV)')
    print(f'\nfix recovers {b-a:+.4f} macro-F1 over the mismatched baseline')
    print(f'CV/LB illusion reproduced: C - A = {c-a:+.4f}')


if __name__ == '__main__':
    main()
