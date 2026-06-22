#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Encoder-Decoder Conv(L)STM/GRU for BEV Occupancy Forecasting
- History: T frames
- Forecast: frames T+1 ... T+n-1 (total n-1 future frames)
- Features:
  * Teacher Forcing / Scheduled Sampling
  * BCE / Dice / Focal losses (+ optional contrast consistency)
  * Dynamic-pixel weighting or static sampling
  * Per-horizon evaluation (IoU / Dynamic IoU / P / R / F1 / Acc)
  * Checkpoint saving every N steps
  * Eval-only mode with optional result export

通信体积优化（新增三步，可独立开关，亦可叠加）:
1) --enable_gru_fp16           : 使用 ConvGRU（只传 h），并对“传输态”做 FP16。
2) --enable_ds64_fp16          : 时序编码在 64×64 分辨率（H/4,W/4）进行，并对“传输态”做 FP16；该选项隐式启用 GRU。
3) --enable_bottleneck_1x1     : 在传输态上加 1×1 通道瓶颈（默认 32→12，可调 --bottleneck_dim）。

注意：
- “传输态”在本脚本中以内联方式模拟（pack→(to half)→unpack），以便端到端训练适配压缩误差；
  真正部署可把 pack 的输出序列化后发送，接收端做解包并接 decoder。
"""

import time
import os, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb
from tqdm import tqdm

LOG_FILE = None

# ========================
# Config & Device
# ========================

def get_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument('--data_root', nargs='+', required=True, help='List of training data root paths')
    p.add_argument('--data_root_val', nargs='+', default=None, help='List of validation data root paths')
    p.add_argument('--T', type=int, default=20, help='history length')
    p.add_argument('--n', type=int, default=10, help='total span; predict n-1 frames')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--eval_interval', type=int, default=50, help='validation frequency (steps)')
    p.add_argument('--max_steps', type=int, default=0,
                   help='Stop after this many optimizer steps; 0 runs all epochs.')
    p.add_argument('--max_val_batches', type=int, default=0,
                   help='Evaluate only this many validation batches; 0 uses the full set.')
    p.add_argument('--use_aug', action='store_true')

    # weighting / curriculum
    p.add_argument('--static_sampling_ratio', type=float, default=1.0,
                   help='Fraction of static pixels to sample (0-1).')
    p.add_argument('--curriculum_epochs', type=int, default=5,
                   help='Epochs before enabling advanced losses')
    p.add_argument('--use_dyn_weight', action='store_true',
                   help='Weight dynamic pixels higher if enabled')
    p.add_argument('--lambda_d', type=float, default=4.0,
                   help='Dynamic-pixel weight factor')

    # heads & losses
    p.add_argument('--use_change_head', action='store_true',
                   help='Predict per-step change map as auxiliary head')
    p.add_argument('--use_dice', action='store_true')
    p.add_argument('--use_focal', action='store_true')
    p.add_argument('--use_contrast', action='store_true',
                   help='Temporal smoothness (MSE between step t pred and decoder input)')
    p.add_argument('--alpha_dice', type=float, default=1.0)
    p.add_argument('--alpha_chg', type=float, default=0.5)
    p.add_argument('--alpha_focal', type=float, default=1.0)
    p.add_argument('--alpha_contrast', type=float, default=0.1)

    # teacher forcing / sampling
    p.add_argument('--teacher_forcing_start', type=float, default=1.0)
    p.add_argument('--teacher_forcing_end', type=float, default=0.0)
    p.add_argument('--teacher_forcing_epochs', type=int, default=10)

    # checkpointing
    p.add_argument('--ckpt_interval', type=int, default=10, help='Save checkpoint every N steps')
    p.add_argument('--ckpt_dir', type=str, default='./checkpoints')

    # eval-only
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--ckpt_path', type=str, help='Path to checkpoint for eval-only mode')
    p.add_argument('--out_dir', type=str, help='Directory to save predictions in eval-only')

    # logging
    p.add_argument('--project', default='bev-encdec')
    p.add_argument('--name', default='run1')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                   help='Device to use. auto selects cuda:0 when available, otherwise cpu.')

    # ========= 通信体积优化：三步开关 =========
    p.add_argument('--enable_gru_fp16', action='store_true',
                   help='Step-1: use ConvGRU (only h) and cast transmission to FP16')
    p.add_argument('--enable_ds64_fp16', action='store_true',
                   help='Step-2: temporal encoder @64x64 (H/4,W/4), cast transmission to FP16 (implicitly enables GRU)')
    p.add_argument('--enable_bottleneck_1x1', action='store_true',
                   help='Step-3: 1x1 conv bottleneck on transmitted state (e.g., 32->12)')
    p.add_argument('--bottleneck_dim', type=int, default=12, help='Channels after 1x1 bottleneck when enabled')

    return p.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def report_device(device: torch.device) -> None:
    print(f"使用设备: {device}")
    try:
        if device.type == "cuda":
            print(f"显卡名称: {torch.cuda.get_device_name(0)}")
            print(f"已保留显存: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
            free_bytes, total_bytes = torch.cuda.mem_get_info(device=torch.cuda.current_device())
            print(f"设备 {torch.cuda.current_device()} 可用显存：{free_bytes/1024**3:.2f} GB / {total_bytes/1024**3:.2f} GB")
    except Exception:
        pass


DEVICE = resolve_device("auto")


# ========================
# Utilities
# ========================

def sparse2bev(idx, shape):
    bev = np.zeros(shape, np.uint8)
    y, x = idx[:, 0], idx[:, 1]
    m = (y >= 0) & (y < shape[0]) & (x >= 0) & (x < shape[1])
    bev[y[m], x[m]] = 1
    return bev

def dice_loss(pred, target, eps=1e-6):
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    inter = (p * t).sum(1)
    union = p.sum(1) + t.sum(1)
    return (1 - (2 * inter + eps) / (union + eps)).mean()

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    p_t = pred * target + (1 - pred) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (-alpha_t * (1 - p_t).pow(gamma) * torch.log(p_t.clamp(min=1e-8))).mean()

def iou_per_sample(pred, true):
    p = pred.cpu().numpy().astype(bool)
    t = true.cpu().numpy().astype(bool)
    inter = np.logical_and(p, t).sum()
    uni = np.logical_or(p, t).sum()
    return inter / uni if uni > 0 else 1.0

def prfa_numpy(pred, true):
    p = precision_score(true.flatten(), pred.flatten(), zero_division=1)
    r = recall_score(true.flatten(), pred.flatten(), zero_division=1)
    f = f1_score(true.flatten(), pred.flatten(), zero_division=1)
    a = (pred.flatten() == true.flatten()).mean()
    return p, r, f, a

def linear_tf_prob(epoch, start, end, total_epochs):
    if epoch <= 1: return start
    if total_epochs <= 1: return end
    ratio = min(1.0, max(0.0, (epoch - 1) / (total_epochs - 1)))
    return start + (end - start) * ratio


def build_combined_dataset(roots: list[str], T: int, n: int, aug: bool = False) -> Dataset:
    datasets = [SeqDataset(root, T, n, aug, H=256, W=256) for root in roots]
    return ConcatDataset(datasets)

# ========================
# Dataset
# ========================

class SeqDataset(Dataset):
    def __init__(self, root, T, n, aug=False, H=256, W=256):
        assert n >= 2
        self.root, self.T, self.n = root, T, n
        self.aug, self.H, self.W = aug, H, W
        self.samples = []
        by_scene = {}
        for path in glob.glob(os.path.join(root, '*_*')):
            sc, fr = map(int, os.path.basename(path).split('_'))
            d = np.load(f'{root}/{sc}_{fr}/0.npy', allow_pickle=True).item()
            if d:
                by_scene.setdefault(sc, []).append(fr)
        for sc, frs in by_scene.items():
            frs.sort()
            L = max(0, len(frs) - T - (n-1) + 1)
            for st in range(L):
                hist = frs[st:st+T]
                tgt  = frs[st+T:st+T+(n-1)]
                self.samples.append((sc, hist, tgt))

    def __len__(self): return len(self.samples)

    def _load_bev(self, sc, fr):
        d = np.load(f'{self.root}/{sc}_{fr}/0.npy', allow_pickle=True).item()
        return sparse2bev(d['voxel_indices_0'], (self.H, self.W))

    def __getitem__(self, idx):
        sc, hist_ids, tgt_ids = self.samples[idx]
        X = [self._load_bev(sc, fr) for fr in hist_ids]
        Y = [self._load_bev(sc, fr) for fr in tgt_ids]
        if self.aug:
            flip = np.random.rand()<0.5; rot = np.random.choice([0,2])
            X = [np.rot90(np.fliplr(x) if flip else x, rot).copy() for x in X]
            Y = [np.rot90(np.fliplr(y) if flip else y, rot).copy() for y in Y]
        X = torch.from_numpy(np.stack(X)).unsqueeze(1).float()
        Y = torch.from_numpy(np.stack(Y)).unsqueeze(1).float()
        assert X.shape[-2:]==(256,256)
        assert Y.shape[-2:]==(256,256)
        return X, Y, sc, hist_ids, tgt_ids

# ========================
# Model: Cells
# ========================

class ConvLSTMCell(nn.Module):
    def __init__(self, c_in, c_hid):
        super().__init__()
        self.c_hid = c_hid
        self.conv = nn.Conv2d(c_in + c_hid, 4 * c_hid, 3, padding=1)

    def forward(self, x, h, c):
        g = self.conv(torch.cat([x, h], dim=1))
        i, f, o, gg = g.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        gg = torch.tanh(gg)
        c2 = f * c + i * gg
        h2 = o * torch.tanh(c2)
        return h2, c2

class ConvGRUCell(nn.Module):
    def __init__(self, c_in, c_hid):
        super().__init__()
        self.conv_zr = nn.Conv2d(c_in + c_hid, 2 * c_hid, 3, padding=1)
        self.conv_h  = nn.Conv2d(c_in + c_hid, c_hid, 3, padding=1)

    def forward(self, x, h):
        z, r = self.conv_zr(torch.cat([x, h], 1)).chunk(2, 1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)
        h_tilde = torch.tanh(self.conv_h(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * h_tilde

# ========================
# Model: Encoders / Decoders
# ========================

class EncoderConvLSTM(nn.Module):
    """ Consume T frames: x[:, t] shape = (B,1,H,W) """
    def __init__(self, c_in=1, c_hid=32):
        super().__init__()
        self.c_hid = c_hid
        self.cell = ConvLSTMCell(c_in, c_hid)

    def forward(self, x):
        B, T, C, H, W = x.shape
        h = x.new_zeros((B, self.c_hid, H, W))
        c = x.new_zeros((B, self.c_hid, H, W))
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
        return h, c

class DecoderConvLSTM(nn.Module):
    def __init__(self, c_in=1, c_hid=32):
        super().__init__()
        self.cell = ConvLSTMCell(c_in, c_hid)

    def forward(self, x0, h0, c0, steps, step_input_fn, step_output_fn):
        h, c = h0, c0
        x_in = x0
        preds, auxes = [], []
        for t in range(steps):
            h, c = self.cell(x_in, h, c)
            pred, aux = step_output_fn(h)
            preds.append(pred)
            auxes.append(aux)
            x_in = step_input_fn(t, pred)
        return preds, auxes

class EncoderConvGRU(nn.Module):
    def __init__(self, c_in=1, c_hid=32):
        super().__init__()
        self.c_hid = c_hid
        self.cell = ConvGRUCell(c_in, c_hid)

    def forward(self, x):
        B, T, C, H, W = x.shape
        h = x.new_zeros((B, self.c_hid, H, W))
        for t in range(T):
            h = self.cell(x[:, t], h)
        return h  # only h

class DecoderConvGRU(nn.Module):
    def __init__(self, c_in=1, c_hid=32):
        super().__init__()
        self.cell = ConvGRUCell(c_in, c_hid)

    def forward(self, x0, h0, steps, step_input_fn, step_output_fn):
        h = h0
        x_in = x0
        preds, auxes = [], []
        for t in range(steps):
            h = self.cell(x_in, h)
            pred, aux = step_output_fn(h)
            preds.append(pred)
            auxes.append(aux)
            x_in = step_input_fn(t, pred)
        return preds, auxes

# ========================
# Model: BEV Forecast Net (with link compression simulation)
# ========================

class BEVForecastNet(nn.Module):
    """
    - Encoder: Conv(L)STM or ConvGRU over T history frames
    - Optional downsampling to 64x64 (H/4, W/4) for temporal encoding
    - Simulate "transmission": pack(1x1)->(optional FP16)->unpack(1x1)
    - Decoder: Conv(L)STM or ConvGRU at full 256x256
    - Heads:
        * occ_head: (B,1,H,W)
        * change_head (optional)
    """
    def __init__(self, args, use_change_head=False, c_hid=32):
        super().__init__()
        self.args = args
        self.c_hid = c_hid

        # Step flags
        self.use_ds64   = bool(args.enable_ds64_fp16)
        # Step-2 implies GRU
        self.use_gru    = bool(args.enable_gru_fp16 or args.enable_ds64_fp16)
        # FP16 is enabled if step-1 or step-2 is on (两步描述都包含 FP16)
        self.use_fp16_link = bool(args.enable_gru_fp16 or args.enable_ds64_fp16)
        self.use_bottleneck = bool(args.enable_bottleneck_1x1)
        self.bottleneck_dim = int(args.bottleneck_dim)

        # Temporal encoders (work at 256x256 or 64x64 depending on self.use_ds64)
        if self.use_gru:
            self.encoder = EncoderConvGRU(1, c_hid)
            self.decoder = DecoderConvGRU(1, 32)
        else:
            self.encoder = EncoderConvLSTM(1, c_hid)
            self.decoder = DecoderConvLSTM(1, 32)

        # Heads at full res
        self.occ_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1), nn.Sigmoid()
        )
        self.use_change_head = use_change_head
        if use_change_head:
            self.chg_head = nn.Sequential(
                nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1), nn.Sigmoid()
            )

        # Pack/Unpack (1x1 bottleneck) configured by whether GRU or LSTM is used
        in_ch = 32 if self.use_gru else 64  # GRU: h; LSTM: [h,c]
        if self.use_bottleneck:
            self.pack_conv   = nn.Conv2d(in_ch, self.bottleneck_dim, kernel_size=1)
            self.unpack_conv = nn.Conv2d(self.bottleneck_dim, in_ch, kernel_size=1)
        else:
            self.pack_conv   = nn.Identity()
            self.unpack_conv = nn.Identity()

        # handy numbers for DS/US
        self._ds_factor = 4  # 256 -> 64

    def _down(self, x):
        """Average-pool downsample to 64x64 if enabled."""
        if not self.use_ds64:
            return x
        return F.avg_pool2d(x, kernel_size=self._ds_factor, stride=self._ds_factor)

    def _up(self, x):
        """Bilinear upsample back to 256x256 if DS64 was used."""
        if not self.use_ds64:
            return x
        return F.interpolate(x, scale_factor=self._ds_factor, mode='bilinear', align_corners=False)

    def _pack_link(self, h, c=None):
        """
        Compose the transmitted tensor S, apply 1x1 bottleneck and (optional) FP16 cast.
        Returns z (possibly half), plus a flag telling whether we used GRU.
        """
        if self.use_gru:
            S = h
        else:
            S = torch.cat([h, c], dim=1)  # (B, 64, H, W)
        z = self.pack_conv(S)
        if self.use_fp16_link:
            z = z.half()
        return z

    def _unpack_link(self, z):
        """
        Inverse of _pack_link: cast to float32, 1x1 expand back, and split if LSTM.
        """
        z = z.float()
        S = self.unpack_conv(z)
        if self.use_gru:
            return S, None
        else:
            h, c = S.chunk(2, dim=1)
            return h, c

    def forward(self, X, steps, step_input_fn, use_change_head=False):
        """
        X: (B,T,1,256,256)
        """
        B, T, C, H, W = X.shape
        # 1) Prepare encoder inputs (optionally downsample each frame to 64x64)
        X_enc = torch.stack([self._down(X[:, t]) for t in range(T)], dim=1)  # (B,T,1,H',W')

        # 2) Temporal encode
        if self.use_gru:
            h0_ds = self.encoder(X_enc)                # (B,32,H',W')
            c0_ds = None
        else:
            h0_ds, c0_ds = self.encoder(X_enc)         # (B,32,H',W'), (B,32,H',W')

        # 3) Simulate "sending" the state through a compressed link
        z = self._pack_link(h0_ds, c0_ds)              # pack (and optional FP16)
        h0_ds_rec, c0_ds_rec = self._unpack_link(z)    # unpack (float32 again)

        # 4) Upsample back to full res for decoder
        h0 = self._up(h0_ds_rec)
        c0 = self._up(c0_ds_rec) if (c0_ds_rec is not None) else None

        # 5) Decoder at full res
        def step_output_fn(h):
            occ = self.occ_head(h)
            chg = self.chg_head(h) if (self.use_change_head and use_change_head) else None
            return occ, chg

        x0 = X[:, -1]  # last observed frame (full res)
        if self.use_gru:
            preds, auxes = self.decoder(x0, h0, steps, step_input_fn, step_output_fn)
        else:
            preds, auxes = self.decoder(x0, h0, c0, steps, step_input_fn, step_output_fn)

        occ_seq = torch.stack(preds, dim=1)  # (B,steps,1,H,W)
        if auxes[0] is None:
            chg_seq = None
        else:
            chg_seq = torch.stack(auxes, dim=1)
        return occ_seq, chg_seq

# ========================
# Evaluation
# ========================

@torch.no_grad()
def evaluate(model, loader, epoch, step, steps, args, use_wandb=True):
    model.eval()

    ious_t       = [[] for _ in range(steps)]
    base_ious_t  = [[] for _ in range(steps)]
    deltas_t     = [[] for _ in range(steps)]
    dyn_ious_t   = [[] for _ in range(steps)]
    P_t, R_t, F_t, A_t = [[] for _ in range(steps)], [[] for _ in range(steps)], [[] for _ in range(steps)], [[] for _ in range(steps)]

    for batch_id, (X, Y, *_) in enumerate(loader):
        if args.max_val_batches and batch_id >= args.max_val_batches:
            break
        X, Y = X.to(DEVICE), Y.to(DEVICE)

        def step_input_fn(_t, prev_pred):
            return prev_pred.detach()

        occ_seq, _ = model(X, steps, step_input_fn, use_change_head=False)
        pred_bin = (occ_seq > 0.5).float()
        base = X[:, -1:].repeat(1, steps, 1, 1, 1)

        B = X.size(0)
        for t in range(steps):
            p_t = pred_bin[:, t, 0]
            y_t = Y[:, t, 0]
            b_t = base[:, t, 0]

            iou_batch = np.mean([iou_per_sample(p_t[b], y_t[b]) for b in range(B)])
            iou_base  = np.mean([iou_per_sample(b_t[b], y_t[b]) for b in range(B)])
            delta     = iou_batch - iou_base

            mask = (b_t != y_t).float()
            di = []
            for b in range(B):
                p_dyn = (p_t[b] * mask[b]).cpu().numpy().astype(bool)
                t_dyn = (y_t[b] * mask[b]).cpu().numpy().astype(bool)
                inter = np.logical_and(p_dyn, t_dyn).sum()
                uni   = np.logical_or (p_dyn, t_dyn).sum()
                di.append(inter / uni if uni > 0 else 1.0)
            dyn_iou_batch = float(np.mean(di))

            ious_t[t].append(iou_batch)
            base_ious_t[t].append(iou_base)
            deltas_t[t].append(delta)
            dyn_ious_t[t].append(dyn_iou_batch)

        for t in range(steps):
            p = pred_bin[:, t].cpu().numpy()
            g = Y[:, t].cpu().numpy()
            pr, rc, f1, acc = prfa_numpy(p, g)
            P_t[t].append(pr); R_t[t].append(rc); F_t[t].append(f1); A_t[t].append(acc)

    per_horizon = []
    for t in range(steps):
        stats = {
            'IoU':          float(np.mean(ious_t[t]))        if ious_t[t]      else float('nan'),
            'Baseline_IoU': float(np.mean(base_ious_t[t]))   if base_ious_t[t] else float('nan'),
            'Delta_IoU':    float(np.mean(deltas_t[t]))      if deltas_t[t]    else float('nan'),
            'Dynamic_IoU':  float(np.mean(dyn_ious_t[t]))    if dyn_ious_t[t]  else float('nan'),
            'Precision':    float(np.mean(P_t[t]))           if P_t[t]         else float('nan'),
            'Recall':       float(np.mean(R_t[t]))           if R_t[t]         else float('nan'),
            'F1':           float(np.mean(F_t[t]))           if F_t[t]         else float('nan'),
            'Accuracy':     float(np.mean(A_t[t]))           if A_t[t]         else float('nan'),
        }
        per_horizon.append(stats)

    overall = {}
    if steps > 0:
        for k in per_horizon[0].keys():
            overall[k] = float(np.mean([ph[k] for ph in per_horizon]))

    if True:
        log_dict = {'epoch': epoch, 'step': step}
        for t, ph in enumerate(per_horizon, start=1):
            log_dict.update({
                f'val/IoU_t{t}':          ph['IoU'],
                f'val/Baseline_IoU_t{t}': ph['Baseline_IoU'],
                f'val/Delta_IoU_t{t}':    ph['Delta_IoU'],
                f'val/Dynamic_IoU_t{t}':  ph['Dynamic_IoU'],
                f'val/Precision_t{t}':    ph['Precision'],
                f'val/Recall_t{t}':       ph['Recall'],
                f'val/F1_t{t}':           ph['F1'],
                f'val/Accuracy_t{t}':     ph['Accuracy'],
            })
        log_dict.update({
            'val/IoU':          overall.get('IoU', float('nan')),
            'val/Baseline_IoU': overall.get('Baseline_IoU', float('nan')),
            'val/Delta_IoU':    overall.get('Delta_IoU', float('nan')),
            'val/Dynamic_IoU':  overall.get('Dynamic_IoU', float('nan')),
            'val/Precision':    overall.get('Precision', float('nan')),
            'val/Recall':       overall.get('Recall', float('nan')),
            'val/F1':           overall.get('F1', float('nan')),
            'val/Accuracy':     overall.get('Accuracy', float('nan')),
        })
    if use_wandb:
        wandb.log(log_dict)
    print(log_dict)

    if steps > 0:
        msg = [f"Epoch {epoch}, Step {step}"]
        for t, ph in enumerate(per_horizon, start=1):
            msg.append(f"[t+{t}] IoU {ph['IoU']:.4f} | ΔIoU {ph['Delta_IoU']:.4f} | DynIoU {ph['Dynamic_IoU']:.4f}")
        msg.append(f"Overall IoU {overall['IoU']:.4f}, ΔIoU {overall['Delta_IoU']:.4f}, DynIoU {overall['Dynamic_IoU']:.4f}")
        print(" | ".join(msg))
        LOG_FILE.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"{ {'epoch': epoch, 'step': step, 'IoU': overall['IoU'], 'Delta_IoU': overall['Delta_IoU'], 'Dynamic_IoU': overall['Dynamic_IoU']} }\n")
        LOG_FILE.flush()
    else:
        print(f"Epoch {epoch}, Step {step} (no future steps)")

    model.train()
    return {'per_horizon': per_horizon, 'overall': overall}

def export_predictions(net, loader, steps, args):
    os.makedirs(args.out_dir, exist_ok=True)

    def iou_numpy(pred_bool, gt_bool):
        inter = np.logical_and(pred_bool, gt_bool).sum()
        uni   = np.logical_or (pred_bool, gt_bool).sum()
        return float(inter / uni) if uni > 0 else 1.0

    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("scene,interval,last_input_frame,target_frame,iou,baseline_iou\n")

    for X, Y, sc, hist_ids, tgt_ids in tqdm(loader, desc="Exporting predictions"):
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)
        B = X.shape[0]

        per_sample_tgt = [[int(tensor[b].item()) for tensor in tgt_ids] for b in range(B)]
        last_obs_ids = [int(hist_ids[-1][b].item()) for b in range(B)]

        def step_input_fn(_t, prev):
            return prev.detach()

        with torch.no_grad():
            occ_seq, _ = net(X, steps, step_input_fn, use_change_head=False)

        preds = (occ_seq > 0.5).cpu().numpy()
        gts   = Y.detach().cpu().numpy()
        base  = X[:, -1:].repeat(1, steps, 1, 1, 1).detach().cpu().numpy()

        for b in range(B):
            sample_sc = int(sc[b])
            folder = os.path.join(args.out_dir, f"{sample_sc}")
            os.makedirs(folder, exist_ok=True)
            input_last_id = last_obs_ids[b]

            for i, tgt_fr in enumerate(per_sample_tgt[b]):
                bev_pred = preds[b, i, 0]
                bev_gt   = gts[b, i, 0] > 0.5
                bev_base = base[b, i, 0] > 0.5

                iou_val       = iou_numpy(bev_pred.astype(bool), bev_gt)
                baseline_iou  = iou_numpy(bev_base, bev_gt)

                idxs = np.argwhere(bev_pred == 1)
                interval = i + 1
                save_path = os.path.join(folder, f"{interval}")
                os.makedirs(save_path, exist_ok=True)

                np.save(os.path.join(save_path, f"{input_last_id}_{tgt_fr}.npy"),
                        {'voxel_indices_0': idxs})

                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(f"{sample_sc},{interval},{input_last_id},{tgt_fr},{iou_val:.6f},{baseline_iou:.6f}\n")

            print(f"[scene {sample_sc}] IoU@t+1..t+{steps} ->",
                  [f"{iou_numpy(preds[b, i, 0].astype(bool), (gts[b, i, 0]>0.5)):.3f}" for i in range(steps)])

# ========================
# Training
# ========================

def train(args):
    assert args.n >= 2, "--n must be >= 2 (we predict n-1 future frames)."
    steps = args.n - 1

    global LOG_FILE
    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.ckpt_dir, 'wandb_logs.txt')
    LOG_FILE = open(log_path, 'a', encoding='utf-8')

    wandb.init(project=args.project, name=args.name, config=vars(args))

    root_val = args.data_root_val or args.data_root
    ds_tr = build_combined_dataset(args.data_root, args.T, args.n, args.use_aug)
    ds_va = build_combined_dataset(root_val, args.T, args.n, False)
    trl = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0)
    val = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0)

    net = BEVForecastNet(args, use_change_head=args.use_change_head).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    step = 0
    for ep in range(1, args.epochs + 1):
        tf_prob = linear_tf_prob(ep, args.teacher_forcing_start,
                                 args.teacher_forcing_end, args.teacher_forcing_epochs)
        pbar = tqdm(enumerate(trl), total=len(trl), desc=f"Epoch {ep}/{args.epochs}")
        for batch_idx, (X, Y, *_) in pbar:
            X, Y = X.to(DEVICE), Y.to(DEVICE)

            Y_prev_pad = torch.cat([X[:, -1:], Y[:, :-1]], dim=1)

            def step_input_fn(t, prev_pred):
                if torch.rand(1, device=X.device).item() < tf_prob:
                    return Y_prev_pad[:, t]  # teacher forcing
                else:
                    return prev_pred.detach()

            occ_seq, chg_seq = net(X, steps, step_input_fn, use_change_head=args.use_change_head)

            loss = 0.0
            for t in range(steps):
                p_t = occ_seq[:, t]       # (B,1,H,W)
                y_t = Y[:, t]             # (B,1,H,W)
                prev_t = Y_prev_pad[:, t] # (B,1,H,W)

                dyn = (prev_t != y_t).float()

                if args.static_sampling_ratio < 1.0:
                    stat = (dyn == 0)
                    idx = torch.nonzero(stat.view(-1), as_tuple=False)
                    keep = int(len(idx) * args.static_sampling_ratio)
                    m = torch.zeros_like(dyn.view(-1))
                    if keep > 0 and len(idx) > 0:
                        m[idx[torch.randperm(len(idx))[:keep]]] = 1
                    m = m.view_as(dyn)
                    weight = 1 + args.lambda_d * dyn * m
                    bce = F.binary_cross_entropy(p_t, y_t, weight=weight)
                elif args.use_dyn_weight:
                    weight = 1 + args.lambda_d * dyn
                    bce = F.binary_cross_entropy(p_t, y_t, weight=weight)
                else:
                    bce = F.binary_cross_entropy(p_t, y_t)

                step_loss = bce

                if ep > args.curriculum_epochs:
                    if args.use_dice:
                        step_loss += args.alpha_dice * dice_loss(p_t, y_t)
                    if args.use_change_head and chg_seq is not None:
                        chg_t = chg_seq[:, t]
                        step_loss += args.alpha_chg * F.binary_cross_entropy(chg_t, dyn)
                    if args.use_focal and args.use_change_head and chg_seq is not None:
                        chg_t = chg_seq[:, t]
                        step_loss += args.alpha_focal * focal_loss(chg_t, dyn)

                if args.use_contrast:
                    step_loss += args.alpha_contrast * F.mse_loss(p_t, prev_t)

                loss = loss + step_loss

            loss = loss / steps

            opt.zero_grad()
            loss.backward()
            opt.step()

            wandb.log({'train/loss': loss.item(), 'epoch': ep, 'step': step, 'tf_prob': tf_prob})
            print(f"Epoch {ep}, Step {step}, Loss {loss.item():.4f}, TF Prob {tf_prob:.4f}")
            pbar.set_postfix(step=step, loss=f"{loss.item():.4f}")

            if step and step % args.eval_interval == 0:
                evaluate(net, val, ep, step, steps, args)
            if step and step % args.ckpt_interval == 0:
                ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step}.pth")
                os.makedirs(args.ckpt_dir, exist_ok=True)
                torch.save({'model': net.state_dict(), 'step': step}, ckpt_path)
                print(f"Checkpoint saved at step {step}")
            step += 1
            if args.max_steps and step >= args.max_steps:
                evaluate(net, val, ep, step, steps, args)
                wandb.finish()
                LOG_FILE.close()
                return

    evaluate(net, val, args.epochs, step, steps, args)
    wandb.finish()
    LOG_FILE.close()

# ========================
# Entry point
# ========================

def main():
    args = get_args()
    global DEVICE
    DEVICE = resolve_device(args.device)
    report_device(DEVICE)
    steps = args.n - 1
    net = BEVForecastNet(args, use_change_head=args.use_change_head).to(DEVICE)
    if args.eval_only:
        ckpt = torch.load(args.ckpt_path, map_location=DEVICE)
        net.load_state_dict(ckpt['model'])
        root_val = args.data_root_val
        ds_va = build_combined_dataset(root_val, args.T, args.n, False)
        val_loader = DataLoader(ds_va, args.batch, False)
        if args.out_dir:
            export_predictions(net, val_loader, steps, args)
        else:
            evaluate(net, val_loader, 0, 0, steps, args, use_wandb=False)
    else:
        train(args)

if __name__=='__main__':
    main()
