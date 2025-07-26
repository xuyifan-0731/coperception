# 完整训练脚本：添加 IoU_baseline 和 ΔIoU 指标，并记录到 wandb

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
parser.add_argument('--T', type=int, default=5)
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
    # pred,true: [B,1,H,W] 二值
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

def augment_bev(bev):
    if np.random.rand() < 0.5:
        bev = np.fliplr(bev)
    k = np.random.choice([0,1,2,3])
    bev = np.rot90(bev, k)
    return bev

# ========== DATASET ==========
class BEVSequenceDataset(Dataset):
    def __init__(self, root_dir, T, use_aug=False):
        self.root = root_dir; self.T = T; self.use_aug = use_aug
        self.by_scene = {}
        for folder in sorted(glob.glob(os.path.join(root_dir, "*_*"))):
            sc, fr = os.path.basename(folder).split("_")
            sc, fr = int(sc), int(fr)
            self.by_scene.setdefault(sc, []).append(fr)
        for sc in self.by_scene: self.by_scene[sc].sort()
        # infer H,W
        for sc, frs in self.by_scene.items():
            if len(frs)>=T+1:
                data = np.load(os.path.join(root_dir, f"{sc}_{frs[0]}", "0.npy"), allow_pickle=True).item()
                idx0 = data["voxel_indices_0"]
                self.H, self.W = idx0[:,1].max()+1, idx0[:,2].max()+1
                break

    def __len__(self):
        return sum(max(0, len(frs)-self.T) for frs in self.by_scene.values())

    def __getitem__(self, idx):
        cum = 0
        for sc, frs in self.by_scene.items():
            L = len(frs)-self.T
            if idx < cum+L:
                start = idx - cum; hist = frs[start:start+self.T]; tgt = frs[start+self.T]
                X, Y = [], None
                for fr in hist:
                    d = np.load(os.path.join(self.root,f"{sc}_{fr}","0.npy"), allow_pickle=True).item()
                    bev = sparse_to_bev(d["voxel_indices_0"], (self.H, self.W))
                    X.append(bev)
                d = np.load(os.path.join(self.root,f"{sc}_{tgt}","0.npy"), allow_pickle=True).item()
                Y = sparse_to_bev(d["voxel_indices_0"], (self.H, self.W))
                if self.use_aug:
                    flip = np.random.rand() < 0.5
                    rot_k = np.random.choice([0, 2])
                    for i in range(len(X)):
                        if flip:
                            X[i] = np.fliplr(X[i])
                        X[i] = np.rot90(X[i], rot_k)
                        X[i] = X[i].copy()  # <—— 这里确保正向内存
                    if flip:
                        Y = np.fliplr(Y)
                    Y = np.rot90(Y, rot_k)
                    Y = Y.copy()  # <—— 这里也要 copy

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
        p=k//2
        self.conv=nn.Conv2d(in_ch+hid_ch,4*hid_ch,k,padding=p)
        self.hid_ch=hid_ch
    def forward(self,x,h,c):
        g=self.conv(torch.cat([x,h],1))
        i,f,o,gc = g.chunk(4,1)
        i,f,o = torch.sigmoid(i),torch.sigmoid(f),torch.sigmoid(o)
        gc=torch.tanh(gc)
        c2 = f*c + i*gc
        h2 = o*torch.tanh(c2)
        return h2,c2


class ConvLSTM(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3, layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvLSTMCell(in_ch if i==0 else hid_ch, hid_ch, k)
            for i in range(layers)
        ])

    def forward(self, x):
        B, T, C, H, W = x.shape
        # 初始化 h 和 c，都使用 x.new_zeros：
        h = [x.new_zeros(B, cell.hid_ch, H, W) for cell in self.layers]
        c = [x.new_zeros(B, cell.hid_ch, H, W) for cell in self.layers]

        for t in range(T):
            inp = x[:, t]
            for i, cell in enumerate(self.layers):
                h[i], c[i] = cell(inp, h[i], c[i])
                inp = h[i]
        return h[-1]

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=ConvLSTM(1,32)
        self.dec=nn.Sequential(
            nn.Conv2d(32,32,3,padding=1),nn.ReLU(),
            nn.Conv2d(32,1,1),nn.Sigmoid()
        )
    def forward(self,x):
        return self.dec(self.enc(x))

# ========== TRAIN / EVAL ==========
def train():
    wandb.init(project=args.project, config=vars(args), name=args.name)
    tr=BEVSequenceDataset(args.data_root,args.T,args.use_augmentation)
    va=BEVSequenceDataset(args.data_root_val,args.T,args.use_augmentation)
    trl=DataLoader(tr,batch_size=args.batch_size,shuffle=True)
    val=DataLoader(va,batch_size=args.batch_size,shuffle=False)
    model=Predictor().to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    crit=nn.BCELoss()
    step=0
    for ep in range(1,args.epochs+1):
        model.train()
        for X,Y in trl:
            X,Y=X.to(DEVICE),Y.to(DEVICE)
            out=model(X)
            loss=crit(out,Y)
            opt.zero_grad();loss.backward();opt.step()
            wandb.log({'train/loss':loss.item(),'epoch':ep,'step':step})
            if step % args.eval_interval == 0 and step > 0:
                model.eval()
                iou_p, iou_b, diou_list, pdyn_list = [], [], [], []
                ps, rs, fs, accs = [], [], [], []

                with torch.no_grad():
                    for Xv, Yv in val:
                        Xv, Yv = Xv.to(DEVICE), Yv.to(DEVICE)
                        outv = model(Xv)
                        pv = (outv > 0.5).float()              # [B,1,H,W]
                        base = Xv[:, -1]                       # baseline
                        pb = (base > 0.5).float()

                        # 1) 全图 IoU & ΔIoU
                        iop = intersection_union_iou(pv, Yv)
                        iob = intersection_union_iou(pb, Yv)
                        iou_p.append(iop)
                        iou_b.append(iob)
                        diou_list.append(iop - iob)

                        # 2) Dynamic IoU —— 只在变化区域计算
                        prev = base                              # [B,1,H,W]
                        M = (prev != Yv).float()                 # 1 表示 t-1→t 变化
                        # 将 pv, Yv 按 M 掩码筛选后，计算 IoU
                        # 转 numpy bool
                        p_dyn = (pv * M).cpu().numpy().astype(bool)
                        t_dyn = (Yv * M).cpu().numpy().astype(bool)
                        # squeeze channel
                        p_dyn = p_dyn[:, 0]
                        t_dyn = t_dyn[:, 0]
                        batch_dyn_ious = []
                        for b in range(p_dyn.shape[0]):
                            inter = np.logical_and(p_dyn[b], t_dyn[b]).sum()
                            union = np.logical_or(p_dyn[b], t_dyn[b]).sum()
                            batch_dyn_ious.append(inter/union if union>0 else 1.0)
                        pdyn_list.append(np.mean(batch_dyn_ious))

                        # 3) 其它指标
                        p, r, f1, ac = compute_metrics(pv.squeeze(1), Yv.squeeze(1))
                        ps.append(p); rs.append(r); fs.append(f1); accs.append(ac)

                # 汇总并 log 到 wandb
                wandb.log({
                    'val/IoU':             np.mean(iou_p),
                    'val/Baseline_IoU':    np.mean(iou_b),
                    'val/Delta_IoU':       np.mean(diou_list),
                    'val/Dynamic_IoU':     np.mean(pdyn_list),
                    'val/Precision':       np.mean(ps),
                    'val/Recall':          np.mean(rs),
                    'val/F1':              np.mean(fs),
                    'val/Accuracy':        np.mean(accs),
                    'epoch':               ep,
                    'step':                step,
                })
                model.train()
            step+=1
    wandb.finish()

if __name__=="__main__":
    train()
