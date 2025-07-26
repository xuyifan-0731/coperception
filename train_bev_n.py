#!/usr/bin/env python
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
import wandb

# ========== ARGPARSE & CONFIG ==========
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--data_root_val', type=str, default=None,
                    help='Validation data root, if None, use same as train data')
parser.add_argument('--T', type=int, default=5,
                    help='Number of history frames')
parser.add_argument('--n', type=int, default=1,
                    help='Steps ahead to predict (n=1 → next frame)')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--eval_interval', type=int, default=100)
parser.add_argument('--use_augmentation', action='store_true')
parser.add_argument('--project', type=str, default='bev-pred')
parser.add_argument('--name', type=str, default='bev-pred-test')
args = parser.parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== UTILS ==========
def sparse_to_bev(indices, bev_shape):
    bev = np.zeros(bev_shape, dtype=np.uint8)
    y, x = indices[:,1], indices[:,2]
    valid = (y>=0)&(y<bev_shape[0])&(x>=0)&(x<bev_shape[1])
    bev[y[valid], x[valid]] = 1
    return bev

def intersection_union_iou(pred, true):
    B = pred.shape[0]
    ious = []
    for b in range(B):
        p = pred[b,0].cpu().numpy().astype(bool)
        t = true[b,0].cpu().numpy().astype(bool)
        inter = np.logical_and(p,t).sum()
        union = np.logical_or(p,t).sum()
        ious.append(inter/union if union>0 else 1.0)
    return np.mean(ious)

def compute_metrics(pred, target):
    B = pred.shape[0]
    ps, rs, fs, accs = [], [], [], []
    for b in range(B):
        p = pred[b].cpu().numpy().flatten()
        t = target[b].cpu().numpy().flatten()
        ps.append(precision_score(t, p, zero_division=1))
        rs.append(recall_score(t, p, zero_division=1))
        fs.append(f1_score(t, p, zero_division=1))
        accs.append((p==t).mean())
    return np.mean(ps), np.mean(rs), np.mean(fs), np.mean(accs)

# ========== DATASET ==========
class BEVSequenceDataset(Dataset):
    def __init__(self, root_dir, T, n, use_aug=False):
        self.root = root_dir
        self.T = T
        self.n = n
        self.use_aug = use_aug
        self.by_scene = {}
        for folder in sorted(glob.glob(os.path.join(root_dir, "*_*"))):
            sc, fr = os.path.basename(folder).split("_")
            sc, fr = int(sc), int(fr)
            self.by_scene.setdefault(sc, []).append(fr)
        for sc in self.by_scene:
            self.by_scene[sc].sort()
        # infer H,W from first valid scene
        for sc, frs in self.by_scene.items():
            if len(frs) >= T + n:
                data = np.load(os.path.join(root_dir, f"{sc}_{frs[0]}", "0.npy"), allow_pickle=True).item()
                idx0 = data["voxel_indices_0"]
                self.H, self.W = idx0[:,1].max()+1, idx0[:,2].max()+1
                break

    def __len__(self):
        total = 0
        for frs in self.by_scene.values():
            total += max(0, len(frs) - self.T - self.n + 1)
        return total

    def __getitem__(self, idx):
        cum = 0
        for sc, frs in self.by_scene.items():
            L = max(0, len(frs) - self.T - self.n + 1)
            if idx < cum + L:
                start = idx - cum
                hist = frs[start : start + self.T]
                tgt  = frs[start + self.T + self.n - 1]  # n steps ahead
                X, Y = [], None
                # load history
                for fr in hist:
                    d = np.load(os.path.join(self.root, f"{sc}_{fr}", "0.npy"), allow_pickle=True).item()
                    bev = sparse_to_bev(d["voxel_indices_0"], (self.H, self.W))
                    X.append(bev)
                # load target
                d = np.load(os.path.join(self.root, f"{sc}_{tgt}", "0.npy"), allow_pickle=True).item()
                Y = sparse_to_bev(d["voxel_indices_0"], (self.H, self.W))
                # augmentation
                if self.use_aug:
                    flip = np.random.rand() < 0.5
                    rot_k = np.random.choice([0, 1, 2, 3])
                    for i in range(len(X)):
                        if flip: X[i] = np.fliplr(X[i])
                        X[i] = np.rot90(X[i], rot_k).copy()
                    if flip: Y = np.fliplr(Y)
                    Y = np.rot90(Y, rot_k).copy()
                # to tensor
                X = torch.from_numpy(np.stack(X, 0)).unsqueeze(1).float()
                Y = torch.from_numpy(Y).unsqueeze(0).float()
                return X, Y
            cum += L
        raise IndexError

# ========== MODEL ==========
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        p = k//2
        self.conv = nn.Conv2d(in_ch + hid_ch, 4*hid_ch, k, padding=p)
        self.hid_ch = hid_ch
    def forward(self, x, h, c):
        g = self.conv(torch.cat([x,h],1))
        i,f,o,gg = g.chunk(4,1)
        i,f,o = torch.sigmoid(i),torch.sigmoid(f),torch.sigmoid(o)
        gg = torch.tanh(gg)
        c2 = f*c + i*gg
        h2 = o * torch.tanh(c2)
        return h2, c2

class ConvLSTM(nn.Module):
    def __init__(self, in_ch=1, hid_ch=32):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hid_ch)
    def forward(self, x):
        B,T,C,H,W = x.shape
        h = x.new_zeros(B,32,H,W)
        c = x.new_zeros(B,32,H,W)
        for t in range(T):
            h, c = self.cell(x[:,t], h, c)
        return h

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = ConvLSTM(1,32)
        self.dec = nn.Sequential(
            nn.Conv2d(32,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,1,1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.dec(self.enc(x))

# ========== TRAIN / EVAL ==========
def train():
    wandb.init(project=args.project, config=vars(args), name=args.name)
    # datasets
    ds_tr = BEVSequenceDataset(args.data_root, args.T, args.n, args.use_augmentation)
    val_root = args.data_root_val or args.data_root
    ds_va = BEVSequenceDataset(val_root, args.T, args.n, False)
    trl = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
    val = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False)

    model = Predictor().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.BCELoss()
    step = 0

    for ep in range(1, args.epochs+1):
        model.train()
        for X, Y in trl:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            out = model(X)
            loss = crit(out, Y)
            opt.zero_grad(); loss.backward(); opt.step()

            wandb.log({'train/loss': loss.item(), 'epoch': ep, 'step': step})
            # periodic eval
            if step and step % args.eval_interval == 0:
                evaluate(model, val, ep, step)
            step += 1

    wandb.finish()

def evaluate(model, val_loader, ep, step):
    model.eval()
    iou_p, iou_b, diou_list, pdyn_list = [], [], [], []
    ps, rs, fs, accs = [], [], [], []
    with torch.no_grad():
        for Xv, Yv in val_loader:
            Xv, Yv = Xv.to(DEVICE), Yv.to(DEVICE)
            outv = model(Xv)
            pv = (outv > 0.5).float()
            base = Xv[:,-1]
            pb   = (base > 0.5).float()

            # IoU & ΔIoU
            iop = intersection_union_iou(pv, Yv)
            iob = intersection_union_iou(pb, Yv)
            iou_p.append(iop); iou_b.append(iob); diou_list.append(iop - iob)

            # Dynamic IoU
            M = (base != Yv).float()
            p_dyn = (pv * M).cpu().numpy().astype(bool)[:,0]
            t_dyn = (Yv* M).cpu().numpy().astype(bool)[:,0]
            dyn_ious = [(np.logical_and(p_dyn[b], t_dyn[b]).sum() /
                         np.logical_or(p_dyn[b], t_dyn[b]).sum()
                         if np.logical_or(p_dyn[b], t_dyn[b]).sum()>0 else 1.0)
                        for b in range(p_dyn.shape[0])]
            pdyn_list.append(np.mean(dyn_ious))

            # Precision/Recall/F1/Acc
            p, r, f1, a = compute_metrics(pv.squeeze(1), Yv.squeeze(1))
            ps.append(p); rs.append(r); fs.append(f1); accs.append(a)

    wandb.log({
        'val/IoU':           np.mean(iou_p),
        'val/Baseline_IoU':  np.mean(iou_b),
        'val/Delta_IoU':     np.mean(diou_list),
        'val/Dynamic_IoU':   np.mean(pdyn_list),
        'val/Precision':     np.mean(ps),
        'val/Recall':        np.mean(rs),
        'val/F1':            np.mean(fs),
        'val/Accuracy':      np.mean(accs),
        'epoch':             ep,
        'step':              step,
    })
    model.train()

if __name__ == "__main__":
    train()
