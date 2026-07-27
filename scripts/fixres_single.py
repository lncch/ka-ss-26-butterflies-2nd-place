import numpy as np, pandas as pd, timm, torch, io, sys
from pathlib import Path
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
R=Path(__file__).resolve().parents[1]; D=R/'data/raw'
DEV=('cuda' if torch.cuda.is_available()
     else 'mps' if torch.backends.mps.is_available() else 'cpu')
def gray(paths, root):
    o=np.empty((len(paths),224,224),np.uint8)
    for i,r in enumerate(paths):
        g=Image.open(Path(root)/r).convert('RGB').convert('L')
        b=io.BytesIO(); g.save(b,'JPEG',quality=95); b.seek(0)
        o[i]=np.asarray(Image.open(b).convert('L'),np.uint8)
    return o
class DS(Dataset):
    def __init__(s,a,z): s.a,s.z=a,z
    def __len__(s): return len(s.a)
    def __getitem__(s,i):
        im=Image.fromarray(s.a[i]).convert('RGB').resize((s.z,s.z),Image.BICUBIC)
        x=np.asarray(im,np.float32)/255.
        return torch.from_numpy(((x-0.449)/0.226).transpose(2,0,1))
@torch.inference_mode()
def pred(m,a,z):
    o=[]
    for x in DataLoader(DS(a,z),batch_size=24,num_workers=0):
        x=x.to(DEV); p=m(x).softmax(1)+m(torch.flip(x,(-1,))).softmax(1)
        o.append((p/2).float().cpu().numpy())
    return np.concatenate(o)
tr=pd.read_csv(D/'train.csv'); te=pd.read_csv(D/'test.csv')
classes=sorted(tr.label.unique()); y=tr.label.map({c:i for i,c in enumerate(classes)}).values
_,va=train_test_split(np.arange(len(tr)),test_size=1200,stratify=y,random_state=61)
m=timm.create_model('convnext_base.fb_in22k_ft_in1k',pretrained=False,num_classes=100)
m.load_state_dict(torch.load(R/'kernel_out/bflycnxtlong1/model0_convnext_base/fold0_raw.pt',map_location='cpu'))
m.eval().to(DEV)
va_arr=gray(tr.path.values[va],D)
for z in (224,256):
    print(f'cb18 val @{z}px: {f1_score(y[va],pred(m,va_arr,z).argmax(1),average="macro"):.5f}',flush=True)
np.save('/tmp/cb18_256_test.npy', pred(m,gray(te.path,D/'test'),256))
print('saved cb18 test @256',flush=True)
