# OUC-CGE Starter Kit (Video Group Engagement)

> Baselines siap jalan untuk **SLOW-ResNet50** & **X3D-S** di **OUC-CGE** (3 kelas: Low/Med/High), plus skrip evaluasi **accuracy / macro-F1 / macro-AUC** & **cross-view / cross-layout**.

---

## 0) Struktur Proyek

```
ouc-cge-starter/
├─ data/
│  ├─ videos/                 # Folder berisi semua video klip (mp4/avi)
│  ├─ ouc_cge_master.csv      # CSV metadata utama (lihat format di bawah)
│  ├─ splits/                 # Folder keluaran split
│  │  ├─ train.csv
│  │  ├─ val.csv
│  │  ├─ test.csv
│  │  ├─ crossview_train.csv / crossview_test.csv
│  │  └─ crosslayout_train.csv / crosslayout_test.csv
├─ configs/
│  └─ default.yaml
├─ requirements.txt
├─ prepare_splits.py
├─ ouccge_dataset.py
├─ models.py
├─ train_video.py
├─ eval_video.py
└─ results_table.md           # Tabel template hasil eksperimen
```

---

## 1) Format CSV Metadata (`data/ouc_cge_master.csv`)

Minimal kolom yang diharapkan:

| video_path      | label  | view  | layout       | split |
| --------------- | ------ | ----- | ------------ | ----- |
| videos/0001.mp4 | Low    | front | checkerboard | train |
| videos/0002.mp4 | Medium | side  | round_table  | val   |
| videos/0003.mp4 | High   | back  | checkerboard | test  |

* **label**: salah satu dari `Low`, `Medium`, `High` (case-sensitive).
* **view**: contoh `front`, `side`, `back`.
* **layout**: contoh `checkerboard`, `round_table` (atau sesuai label sebenarnya).
* **split**: `train` / `val` / `test` (bisa diabaikan jika ingin membuat ulang via `prepare_splits.py`).

> **Catatan:** Jika nama kolom di dataset asli berbeda, silakan sesuaikan di `prepare_splits.py` dan `ouccge_dataset.py`.

---

## 2) requirements.txt

```
torch==2.3.1
torchvision==0.18.1
pytorchvideo==0.1.5
pandas==2.2.2
scikit-learn==1.5.2
torchmetrics==1.4.0
pyyaml==6.0.1
opencv-python==4.10.0.84
numpy==1.26.4
tqdm==4.66.4
```

> Pastikan CUDA sesuai (opsional). Jika terjadi konflik versi, gunakan pasangan PyTorch/torchvision yang cocok dengan CUDA lokal Anda.

---

## 3) Config contoh (`configs/default.yaml`)

```yaml
# Data
csv_master: "data/ouc_cge_master.csv"
train_csv: "data/splits/train.csv"
val_csv:   "data/splits/val.csv"
test_csv:  "data/splits/test.csv"
video_root: "data/"         # awalan untuk path video relatif, jika perlu

# Sampling
fps: 4               # sampling 4 fps seperti rekomendasi low-frequency cue
clip_len: 32         # 32 frame ≈ 8 detik @4fps (paper pakai ~10 detik)
resize_short: 256    # resize pendek -> 256
crop_size: 224       # center/ random crop 224

# Train
epochs: 50
batch_size: 8
num_workers: 4
lr: 3e-4
weight_decay: 1e-4

# Model: slow_r50 | x3d_s
model_name: slow_r50
num_classes: 3

# Misc
seed: 42
save_dir: "checkpoints/slow_r50/"
```

---

## 4) `prepare_splits.py`

```python
import pandas as pd
from pathlib import Path

MASTER = Path("data/ouc_cge_master.csv")
OUTDIR = Path("data/splits"); OUTDIR.mkdir(parents=True, exist_ok=True)

# Ganti ini jika kolom Anda berbeda
COL_VIDEO = "video_path"; COL_LABEL = "label"; COL_VIEW = "view"; COL_LAYOUT = "layout"; COL_SPLIT = "split"

assert MASTER.exists(), f"CSV tidak ditemukan: {MASTER}"
df = pd.read_csv(MASTER)

# Jika kolom split belum ada, buat default 80/10/10 per label
if COL_SPLIT not in df.columns:
    dfs = []
    for label, g in df.groupby(COL_LABEL):
        n = len(g)
        n_train = int(0.8 * n); n_val = int(0.1 * n)
        g = g.sample(frac=1.0, random_state=42).reset_index(drop=True)
        g.loc[:n_train-1, COL_SPLIT] = 'train'
        g.loc[n_train:n_train+n_val-1, COL_SPLIT] = 'val'
        g.loc[n_train+n_val:, COL_SPLIT] = 'test'
        dfs.append(g)
    df = pd.concat(dfs, ignore_index=True)

# Simpan split standar
for split in ['train','val','test']:
    df[df[COL_SPLIT]==split].to_csv(OUTDIR/f"{split}.csv", index=False)

# Buat split cross-view: train pada 'front', test pada 'side'+'back' (contoh)
front = df[df[COL_VIEW]=='front']
others = df[df[COL_VIEW].isin(['side','back'])]
front.assign(**{COL_SPLIT:'train'}).to_csv(OUTDIR/"crossview_train.csv", index=False)
others.assign(**{COL_SPLIT:'test'}).to_csv(OUTDIR/"crossview_test.csv", index=False)

# Buat split cross-layout: train pada 'checkerboard', test pada 'round_table' (contoh)
check = df[df[COL_LAYOUT]=='checkerboard']
roundt = df[df[COL_LAYOUT]=='round_table']
check.assign(**{COL_SPLIT:'train'}).to_csv(OUTDIR/"crosslayout_train.csv", index=False)
roundt.assign(**{COL_SPLIT:'test'}).to_csv(OUTDIR/"crosslayout_test.csv", index=False)

print("Selesai menulis splits di data/splits/")
```

---

## 5) `ouccge_dataset.py`

```python
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchvision
import torch.nn.functional as F
from torchvision.io import read_video
import random
from pathlib import Path

LABEL_TO_INT = {"Low":0, "Medium":1, "High":2}

class OUC_CGE_VideoDataset(Dataset):
    def __init__(self, csv_path: str, video_root: str = "", fps: int = 4, clip_len: int = 32,
                 resize_short: int = 256, crop_size: int = 224, is_train: bool = True):
        self.df = pd.read_csv(csv_path)
        self.video_root = Path(video_root)
        self.fps = fps
        self.clip_len = clip_len
        self.resize_short = resize_short
        self.crop_size = crop_size
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def _center_crop(self, frames):
        _, h, w = frames.shape[1:]
        if h < self.crop_size or w < self.crop_size:
            scale = max(self.crop_size/h, self.crop_size/w)
            new_h, new_w = int(h*scale), int(w*scale)
            frames = F.interpolate(frames, size=(new_h, new_w), mode='bilinear', align_corners=False)
            _, h, w = frames.shape[1:]
        top = (h - self.crop_size)//2; left = (w - self.crop_size)//2
        return frames[:, :, top:top+self.crop_size, left:left+self.crop_size]

    def _random_crop(self, frames):
        _, h, w = frames.shape[1:]
        if h < self.crop_size or w < self.crop_size:
            scale = max(self.crop_size/h, self.crop_size/w)
            new_h, new_w = int(h*scale), int(w*scale)
            frames = F.interpolate(frames, size=(new_h, new_w), mode='bilinear', align_corners=False)
            _, h, w = frames.shape[1:]
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        return frames[:, :, top:top+self.crop_size, left:left+self.crop_size]

    def _resize_short_side(self, frames):
        _, h, w = frames.shape[1:]
        short = min(h, w)
        scale = self.resize_short/short
        new_h, new_w = int(h*scale), int(w*scale)
        return F.interpolate(frames, size=(new_h, new_w), mode='bilinear', align_corners=False)

    def _temporal_subsample(self, vid_frames: torch.Tensor, vid_fps: float):
        # vid_frames: (T, H, W, C)
        T = vid_frames.shape[0]
        # target_len = seconds = clip_len / fps_target
        # Kita pilih subsampling uniform menjadi clip_len frame sepanjang durasi video
        if T < self.clip_len:
            # loop pad frames
            repeat = (self.clip_len + T - 1)//T
            vid_frames = vid_frames.repeat(repeat, 1, 1, 1)[:self.clip_len]
            T = vid_frames.shape[0]
        idx = torch.linspace(0, T-1, steps=self.clip_len).long()
        return vid_frames.index_select(0, idx)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vpath = str(self.video_root / row['video_path']) if self.video_root else row['video_path']
        label = LABEL_TO_INT[str(row['label'])]
        # read_video -> (T, H, W, C), audio
        video, _, info = read_video(vpath, pts_unit='sec')
        # normalize to float tensor [0,1]
        video = video.float() / 255.0
        # (T, H, W, C) -> (T, C, H, W)
        video = video.permute(0, 3, 1, 2)
        # temporal subsample to clip_len
        video = self._temporal_subsample(video, info.get('video_fps', 30.0))
        # resize short side & crop
        video = self._resize_short_side(video)
        video = self._random_crop(video) if self.is_train else self._center_crop(video)
        # Normalize ImageNet mean/std per frame
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        video = (video - mean) / std
        # (T, C, H, W) -> (C, T, H, W) agar cocok dengan beberapa model video
        video = video.permute(1,0,2,3)
        return video, label
```

---

## 6) `models.py`

```python
import torch
import torch.nn as nn

try:
    from pytorchvideo.models.hub import slow_r50, x3d_s
except Exception as e:
    slow_r50 = None; x3d_s = None

class Identity(nn.Module):
    def forward(self, x):
        return x

def build_model(name: str, num_classes: int):
    name = name.lower()
    if name == 'slow_r50':
        assert slow_r50 is not None, "pytorchvideo tidak tersedia. Install pytorchvideo."
        model = slow_r50(pretrained=True)
        # Ganti head ke num_classes
        if hasattr(model, 'blocks'):
            # pytorchvideo slow_r50 uses model.blocks[5].proj
            model.blocks[5].proj = nn.Linear(model.blocks[5].proj.in_features, num_classes)
        return model
    elif name == 'x3d_s':
        assert x3d_s is not None, "pytorchvideo tidak tersedia. Install pytorchvideo."
        model = x3d_s(pretrained=True)
        # Ganti head
        if hasattr(model, 'blocks'):
            model.blocks[5].proj = nn.Linear(model.blocks[5].proj.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Model {name} belum didukung. Pilih: slow_r50 | x3d_s")
```

---

## 7) `train_video.py`

```python
import argparse, os, yaml, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from ouccge_dataset import OUC_CGE_VideoDataset
from models import build_model


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, crit, opt, device):
    model.train()
    losses = []
    for x, y in tqdm(loader, desc='train', leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward(); opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def eval_epoch(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc='eval', leave=False):
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except Exception:
        auc = float('nan')
    return acc, f1, auc


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get('seed', 42))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Datasets
    train_ds = OUC_CGE_VideoDataset(cfg['train_csv'], cfg.get('video_root',''), cfg['fps'], cfg['clip_len'], cfg['resize_short'], cfg['crop_size'], True)
    val_ds   = OUC_CGE_VideoDataset(cfg['val_csv'],   cfg.get('video_root',''), cfg['fps'], cfg['clip_len'], cfg['resize_short'], cfg['crop_size'], False)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)

    # Model
    model = build_model(cfg['model_name'], cfg['num_classes']).to(device)

    # Optim & loss
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    os.makedirs(cfg['save_dir'], exist_ok=True)
    best_f1 = -1

    for epoch in range(1, cfg['epochs']+1):
        tr_loss = train_one_epoch(model, train_loader, crit, opt, device)
        acc, f1, auc = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch:03d} | loss {tr_loss:.4f} | Acc {acc:.4f} | F1 {f1:.4f} | AUC {auc:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({'model': model.state_dict(), 'cfg': cfg}, os.path.join(cfg['save_dir'], 'best.pt'))
            print("✔ Saved best checkpoint")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='configs/default.yaml')
    args = ap.parse_args()
    main(args.cfg)
```

---

## 8) `eval_video.py`

```python
import argparse, yaml, torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd

from ouccge_dataset import OUC_CGE_VideoDataset
from models import build_model


def evaluate(checkpoint_path: str, cfg_path: str, csv_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = OUC_CGE_VideoDataset(csv_path, cfg.get('video_root',''), cfg['fps'], cfg['clip_len'], cfg['resize_short'], cfg['crop_size'], False)
    dl = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    model = build_model(cfg['model_name'], cfg['num_classes']).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model']); model.eval()

    y_true, y_pred, y_prob = [], [], []
    import torch.nn.functional as F
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_pred.extend(probs.argmax(axis=1).tolist())
            y_true.extend(y.numpy().tolist())

    labels = ['Low','Medium','High']
    report = classification_report(y_true, y_pred, target_names=labels, digits=4, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except Exception:
        auc = float('nan')

    cm = confusion_matrix(y_true, y_pred)
    print("\nClassification Report:\n", report)
    print("Macro AUC:", auc)
    print("Confusion Matrix:\n", pd.DataFrame(cm, index=labels, columns=labels))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cfg', default='configs/default.yaml')
    ap.add_argument('--csv', default='data/splits/test.csv')
    args = ap.parse_args()
    evaluate(args.ckpt, args.cfg, args.csv)
```

---

## 9) Perintah Cepat (How to Run)

```bash
# 1) Siapkan env
pip install -r requirements.txt

# 2) Buat split
python prepare_splits.py

# 3) Latih baseline SLOW-ResNet50
python train_video.py --cfg configs/default.yaml
# (opsional) Latih X3D-S: ganti model_name jadi x3d_s di configs/default.yaml

# 4) Evaluasi di test split standar
python eval_video.py --ckpt checkpoints/slow_r50/best.pt --cfg configs/default.yaml --csv data/splits/test.csv

# 5) Evaluasi Cross-View
python eval_video.py --ckpt checkpoints/slow_r50/best.pt --cfg configs/default.yaml --csv data/splits/crossview_test.csv

# 6) Evaluasi Cross-Layout
python eval_video.py --ckpt checkpoints/slow_r50/best.pt --cfg configs/default.yaml --csv data/splits/crosslayout_test.csv
```

---

## 10) Template Tabel Hasil (copy ke laporan)

Lembar ini (`results_table.md`) siap diisi setelah eksperimen.

```md
### OUC-CGE — Baselines (10 detik, 4 fps, 224x224)
| Model | Params | GFLOPs (≈) | Acc | Macro-F1 | Macro-AUC | Catatan |
|---|---:|---:|---:|---:|---:|---|
| SLOW-R50 | ~32M | ~65 |  |  |  | cfg default.yaml |
| X3D-S    | ~3.8M | ~5.2 |  |  |  | efisien |

### Cross-View (train=front, test=side+back)
| Model | Acc | Macro-F1 | Macro-AUC | ΔAcc vs test-std | Catatan |
|---|---:|---:|---:|---:|---|
| SLOW-R50 |  |  |  |  | generalisasi sudut |
| X3D-S    |  |  |  |  |  |

### Cross-Layout (train=checkerboard, test=round_table)
| Model | Acc | Macro-F1 | Macro-AUC | ΔAcc vs test-std | Catatan |
|---|---:|---:|---:|---:|---|
| SLOW-R50 |  |  |  |  | generalisasi tata ruang |
| X3D-S    |  |  |  |  |  |
```

---

## 11) Catatan & Tips

* **Low-frequency bias**: karena OUC-CGE didominasi sinyal frekuensi rendah, sampling **4 fps** dan klip **8–12 detik** efektif. (Sesuai paper dataset.)
* **Ordinal & kalibrasi (opsional S2)**: ubah loss ke *ordinal regression* atau tambahkan evaluasi **ECE/NLL** untuk kalibrasi keputusan (kode dapat ditambahkan di `train_video.py`).
* **Multi-view fusion (lanjutan)**: buat *late fusion* antar view dengan rata-rata probabilitas dari model yang sama, atau latih model khusus per view.
* **Periksa kolom CSV**: sesuaikan nama kolom jika berbeda dari contoh di atas.
* **Windows tip**: `torchvision.io.read_video` membutuhkan FFmpeg (sudah termasuk di wheel resmi). Jika ada error codec, bisa fallback ke `decord` atau `PyAV`.
