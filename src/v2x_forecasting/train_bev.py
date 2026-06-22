#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Encoder-Decoder ConvLSTM for BEV Occupancy Forecasting
- History: T frames
- Forecast: frames T+1 ... T+n-1 (total n-1 future frames)
- Features:
  * Teacher Forcing / Scheduled Sampling
  * BCE / Dice / Focal losses (+ optional contrast consistency)
  * Dynamic-pixel weighting or static sampling
  * Per-horizon evaluation (IoU / Dynamic IoU / P / R / F1 / Acc)
  * Checkpoint saving every N steps
  * Eval-only mode with optional result export
"""
LOG_FILE = None
import time
import os, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb
from tqdm import tqdm
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
    p.add_argument('--eval_interval', type=int, default=50) #控制“测试”的频率
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
    p.add_argument('--ckpt_interval', type=int, default=10,
                   help='Save checkpoint every N steps') #控制“存模型”的频率
    p.add_argument('--ckpt_dir', type=str, default='./checkpoints') #存模型文件到哪里

    # eval-only
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--ckpt_path', type=str,
                   help='Path to checkpoint for eval-only mode') #指定某个已经训练好的模型 checkpoint 路径，用于评估（或恢复训练）。
    p.add_argument('--out_dir', type=str,
                   help='Directory to save predictions in eval-only') #指定评估结果保存路径

    # logging
    p.add_argument('--project', default='bev-encdec')
    p.add_argument('--name', default='run1')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                   help='Device to use. auto selects cuda:0 when available, otherwise cpu.')
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
    if device.type == "cuda":
        print(f"显卡名称: {torch.cuda.get_device_name(0)}")
        print(f"剩余显存: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device=torch.cuda.current_device())
        free_gb = free_bytes / 1024**3
        total_gb = total_bytes / 1024**3
        print(f"设备 {torch.cuda.current_device()} 可用显存：{free_gb:.2f} GB / {total_gb:.2f} GB")


DEVICE = resolve_device("auto")


# ========================
# Utilities
# ========================

def sparse2bev(idx, shape):
    #import pdb; pdb.set_trace()
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
    print("roots",roots)
    datasets = [SeqDataset(root, T, n, aug,H=256,W=256) for root in roots]
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
        by_scene = {}   # 按场景分组，没有跨场景
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
# Model
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

class EncoderConvLSTM(nn.Module):
    """ Consume T frames: x[:, t] shape = (B,1,H,W) """
    def __init__(self, c_in=1, c_hid=32):
        super().__init__()
        self.cell = ConvLSTMCell(c_in, c_hid)

    def forward(self, x):
        B, T, C, H, W = x.shape
        h = x.new_zeros((B, 32, H, W))
        c = x.new_zeros((B, 32, H, W))
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
        return h, c

class DecoderConvLSTM(nn.Module):
    """
    Autoregressive decode for (n-1) steps.
    Inputs:
      - h0, c0: from encoder
      - x0: first decoder input, typically last observed frame (B,1,H,W)
      - steps: n-1
    """
    def __init__(self, c_in=1, c_hid=32):
        super().__init__()
        self.cell = ConvLSTMCell(c_in, c_hid)

    def forward(self, x0, h0, c0, steps, step_input_fn, step_output_fn):
        """
        step_input_fn(t, prev_pred) -> decoder input at step t (B,1,H,W)
        step_output_fn(h) -> frame prediction (B,1,H,W) and optional aux
        Returns:
          preds: list of (B,1,H,W)
          auxes: list of aux tensors or None
        """
        h, c = h0, c0
        x_in = x0
        preds, auxes = [], []
        for t in range(steps):
            h, c = self.cell(x_in, h, c)
            pred, aux = step_output_fn(h)
            preds.append(pred)
            auxes.append(aux)
            x_in = step_input_fn(t, pred)  # may be GT or previous pred (teacher forcing)
        return preds, auxes

class BEVForecastNet(nn.Module):
    """
    - Encoder: ConvLSTM over T history frames
    - Decoder: ConvLSTM, autoregressive n-1 steps
    - Heads:
        * occ_head: (B,1,H,W) at each step
        * change_head (optional): (B,1,H,W) at each step (predict dynamic regions)
    """
    def __init__(self, use_change_head=False):
        super().__init__()
        self.encoder = EncoderConvLSTM(1, 32)
        self.decoder = DecoderConvLSTM(1, 32)
        self.occ_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1), nn.Sigmoid()
        )
        self.use_change_head = use_change_head
        if use_change_head:
            self.chg_head = nn.Sequential(
                nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 1, 1), nn.Sigmoid()
            )

    def forward(self, X, steps, step_input_fn, use_change_head=False):
        """
        X: (B,T,1,H,W)
        steps: n-1
        step_input_fn: callable for decoder input per step
        Returns:
          occ_seq: (B, steps, 1, H, W)
          chg_seq or None: (B, steps, 1, H, W)
        """
        h0, c0 = self.encoder(X)

        def step_output_fn(h):
            occ = self.occ_head(h)
            chg = self.chg_head(h) if (self.use_change_head and use_change_head) else None
            return occ, chg

        x0 = X[:, -1]  # last observed
        preds, auxes = self.decoder(x0, h0, c0, steps, step_input_fn, step_output_fn)
        occ_seq = torch.stack(preds, dim=1)  # (B,steps,1,H,W)
        if auxes[0] is None:
            chg_seq = None
        else:
            chg_seq = torch.stack(auxes, dim=1)  # (B,steps,1,H,W)
        return occ_seq, chg_seq

# ========================
# Evaluation
# ========================

@torch.no_grad()
def evaluate(model, loader, epoch, step, steps, args, use_wandb=True):
    """
    当 steps = n-1 > 0 时：
      - 逐间隔（t=1..steps）输出/记录各项指标
      - 也保留所有间隔的整体平均指标
    """
    model.eval()

    # --- 按间隔收集：list[steps]，每个元素里是“跨 batch 的采样值列表” ---
    ious_t       = [[] for _ in range(steps)]
    base_ious_t  = [[] for _ in range(steps)]
    deltas_t     = [[] for _ in range(steps)]
    dyn_ious_t   = [[] for _ in range(steps)]
    P_t, R_t, F_t, A_t = [[] for _ in range(steps)], [[] for _ in range(steps)], [[] for _ in range(steps)], [[] for _ in range(steps)]

    for batch_id, (X, Y, *_) in enumerate(loader):
        if args.max_val_batches and batch_id >= args.max_val_batches:
            break
        X, Y = X.to(DEVICE), Y.to(DEVICE)

        # 评估阶段：不使用 Teacher Forcing
        def step_input_fn(_t, prev_pred):
            return prev_pred.detach()

        occ_seq, _ = model(X, steps, step_input_fn, use_change_head=False)   # (B,steps,1,H,W)
        pred_bin = (occ_seq > 0.5).float()
        base = X[:, -1:].repeat(1, steps, 1, 1, 1)

        B = X.size(0)
        # —— 每个未来间隔的 IoU / Baseline IoU / Delta / Dynamic IoU
        for t in range(steps):
            p_t = pred_bin[:, t, 0]
            y_t = Y[:, t, 0]
            b_t = base[:, t, 0]

            # IoU / Baseline IoU（先对 batch 求平均）
            iou_batch = np.mean([iou_per_sample(p_t[b], y_t[b]) for b in range(B)])
            iou_base  = np.mean([iou_per_sample(b_t[b], y_t[b]) for b in range(B)])
            delta     = iou_batch - iou_base

            # Dynamic IoU（仅在 baseline 与 GT 不同的像素上）
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

        # —— 每个未来间隔的 PRFA（把 B,H,W 拉平后算一次，再跨间隔收集）
        for t in range(steps):
            p = pred_bin[:, t].cpu().numpy()
            g = Y[:, t].cpu().numpy()
            pr, rc, f1, acc = prfa_numpy(p, g)
            P_t[t].append(pr); R_t[t].append(rc); F_t[t].append(f1); A_t[t].append(acc)

    # ==== 汇总（按间隔求均值）====
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

    # ==== 同时给出整体平均（跨所有间隔再平均）====
    overall = {}
    if steps > 0:
        for k in per_horizon[0].keys():
            overall[k] = float(np.mean([ph[k] for ph in per_horizon]))

    # ==== 记录到 wandb ====
    if True:
        log_dict = {'epoch': epoch, 'step': step}
        # 每个间隔单独打点：t 从 1 开始更直观
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
        # 总体均值也保留
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

    # ==== 控制台输出（简洁展示）====
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

    # 回到训练模式
    model.train()

    # 可选：把结果返回，便于外部保存/调试
    return {'per_horizon': per_horizon, 'overall': overall}
def export_predictions(net, loader, steps, args):
    """
    Export predictions to .npy files with original sparse format,
    and compute IoU vs. ground truth for each (sample, interval).
    """
    os.makedirs(args.out_dir, exist_ok=True)

    # 小工具：numpy 版 IoU
    def iou_numpy(pred_bool, gt_bool):
        inter = np.logical_and(pred_bool, gt_bool).sum()
        uni   = np.logical_or (pred_bool, gt_bool).sum()
        return float(inter / uni) if uni > 0 else 1.0

    # 结果文件
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("scene,interval,last_input_frame,target_frame,iou,baseline_iou\n")

    for X, Y, sc, hist_ids, tgt_ids in tqdm(loader, desc="Exporting predictions"):
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)  # (B, steps, 1, H, W)
        B = X.shape[0]

        # 将 DataLoader collate 后的 tgt_ids（长度=steps，每个是形状(B,)的tensor）重排为 per-sample 列表
        per_sample_tgt = [[int(tensor[b].item()) for tensor in tgt_ids] for b in range(B)]
        # 同理，取每个样本的最后观测帧 id（hist_ids 同样是长度=T，每个是形状(B,)的tensor）
        last_obs_ids = [int(hist_ids[-1][b].item()) for b in range(B)]

        # 评估阶段：始终用上一步预测作为下一步输入
        def step_input_fn(_t, prev):
            return prev.detach()

        with torch.no_grad():
            occ_seq, _ = net(X, steps, step_input_fn, use_change_head=False)  # (B, steps, 1, H, W)

        preds = (occ_seq > 0.5).cpu().numpy()        # (B, steps, 1, H, W)
        gts   = Y.detach().cpu().numpy()             # (B, steps, 1, H, W)
        base  = X[:, -1:].repeat(1, steps, 1, 1, 1)  # baseline: 最后一帧历史
        base  = base.detach().cpu().numpy()

        for b in range(B):
            sample_sc = int(sc[b])
            folder = os.path.join(args.out_dir, f"{sample_sc}")
            os.makedirs(folder, exist_ok=True)

            # 注意：这里 last_frame 是“模型输入的最后一帧 id”
            input_last_id = last_obs_ids[b]


            for i, tgt_fr in enumerate(per_sample_tgt[b]):
                bev_pred = preds[b, i, 0]             # (H, W) 0/1
                bev_gt   = gts[b, i, 0] > 0.5         # (H, W) -> bool
                bev_base = base[b, i, 0] > 0.5

                # IoU 与 baseline IoU
                iou_val       = iou_numpy(bev_pred.astype(bool), bev_gt)
                baseline_iou  = iou_numpy(bev_base, bev_gt)

                # 存稀疏索引
                idxs = np.argwhere(bev_pred == 1)

                # 文件夹结构：场景/预测间隔/输入最后一帧_目标帧.npy
                interval = i + 1
                save_path = os.path.join(folder, f"{interval}")
                os.makedirs(save_path, exist_ok=True)

                # 约定：文件名里的“输入最后一帧”用真正的输入最后帧 id（而不是反推）
                np.save(os.path.join(save_path, f"{input_last_id}_{tgt_fr}.npy"),
                        {'voxel_indices_0': idxs})

                # 记录指标
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(f"{sample_sc},{interval},{input_last_id},{tgt_fr},{iou_val:.6f},{baseline_iou:.6f}\n")

            # 可选：也可以在这里 print 一些摘要
            print(f"[scene {sample_sc}] IoU@t+1..t+{steps} ->",
                  [f"{iou_numpy(preds[b, i, 0].astype(bool), (gts[b, i, 0]>0.5)):.3f}" for i in range(steps)])







# ========================
# Training
# ========================

def train(args):
    assert args.n >= 2, "--n must be >= 2 (we predict n-1 future frames)."
    steps = args.n - 1  # number of future frames

    global LOG_FILE
    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.ckpt_dir, 'wandb_logs.txt')
    LOG_FILE = open(log_path, 'a', encoding='utf-8')

    wandb.init(project=args.project, name=args.name, config=vars(args))

    root_val = args.data_root_val or args.data_root
    ds_tr = build_combined_dataset(args.data_root, args.T, args.n, args.use_aug)
    #ds_tr = SeqDataset(args.data_root, args.T, args.n, aug=args.use_aug)
    ds_va = build_combined_dataset(root_val, args.T, args.n, False)
    #ds_va = SeqDataset(root_val, args.T, args.n, aug=False)
    trl = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0)
    val = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0)

    net = BEVForecastNet(use_change_head=args.use_change_head).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    step = 0
    for ep in range(1, args.epochs + 1):
        tf_prob = linear_tf_prob(ep, args.teacher_forcing_start,
                                 args.teacher_forcing_end, args.teacher_forcing_epochs)
        pbar = tqdm(enumerate(trl), total=len(trl), desc=f"Epoch {ep}/{args.epochs}")
        #for X, Y, *_ in trl:
        for batch_idx, (X, Y, *_) in pbar:
            X, Y = X.to(DEVICE), Y.to(DEVICE)  # X: (B,T,1,H,W), Y: (B,steps,1,H,W)

            # --- define decoder input policy (teacher forcing with prob tf_prob) ---
            # for loss weighting & change head, we also need a notion of "previous frame" at each step.
            # We'll keep a GT cache to compute dynamic masks consistently.
            Y_prev_pad = torch.cat([X[:, -1:], Y[:, :-1]], dim=1)  # (B,steps,1,H,W), prev GT at each step

            def step_input_fn(t, prev_pred):
                # with prob tf_prob use GT[t-1], else use previous prediction
                if torch.rand(1).item() < tf_prob:
                    return Y_prev_pad[:, t]  # teacher forcing
                else:
                    return prev_pred.detach()

            occ_seq, chg_seq = net(X, steps, step_input_fn, use_change_head=args.use_change_head)
            # occ_seq: (B,steps,1,H,W)

            # --- losses per step ---
            loss = 0.0
            for t in range(steps):
                p_t = occ_seq[:, t]       # (B,1,H,W)
                y_t = Y[:, t]             # (B,1,H,W)
                prev_t = Y_prev_pad[:, t] # (B,1,H,W) previous GT frame for step t

                # dynamic mask (where prev != target at step t)
                dyn = (prev_t != y_t).float()

                # BCE with optional dynamic weighting / static sampling
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

                # curriculum: enable advanced losses after some epochs
                if ep > args.curriculum_epochs:
                    if args.use_dice:
                        step_loss += args.alpha_dice * dice_loss(p_t, y_t)
                    if args.use_change_head and chg_seq is not None:
                        chg_t = chg_seq[:, t]
                        # change target is dyn itself
                        step_loss += args.alpha_chg * F.binary_cross_entropy(chg_t, dyn)
                    if args.use_focal and args.use_change_head and chg_seq is not None:
                        chg_t = chg_seq[:, t]
                        step_loss += args.alpha_focal * focal_loss(chg_t, dyn)

                # optional temporal contrast: encourage p_t close to decoder input (smoothness)
                if args.use_contrast:
                    # decoder input at step t is either prev GT or prev pred (but we can't easily requery here)
                    # use prev_t (GT) as a stable anchor for smoothness
                    step_loss += args.alpha_contrast * F.mse_loss(p_t, prev_t)

                loss = loss + step_loss

            loss = loss / steps

            opt.zero_grad()
            loss.backward()
            opt.step()

            wandb.log({'train/loss': loss.item(), 'epoch': ep, 'step': step, 'tf_prob': tf_prob})
            #LOG_FILE.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                           #f"{ {'train/loss': loss.item(), 'epoch': ep, 'step': step, 'tf_prob': tf_prob} }\n")
            #LOG_FILE.flush()
            #print(f"Epoch {ep}/{args.epochs}, Step {step}, Loss: {loss.item():.4f}, ")
            pbar.set_postfix(step=step, loss=loss.item())


            if step and step % args.eval_interval == 0:
                evaluate(net, val, ep, step, steps, args)
            if step and step % args.ckpt_interval == 0:
                '''
                torch.save({'model': net.state_dict(), 'step': step},
                           os.path.join(args.ckpt_dir, f"ckpt_{step}.pth"))
                '''
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

    # final eval
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
    net = BEVForecastNet(use_change_head=args.use_change_head).to(DEVICE)
    if args.eval_only:
        ckpt = torch.load(args.ckpt_path, map_location=DEVICE)
        #import pdb; pdb.set_trace()
        net.load_state_dict(ckpt['model'])
        root_val = args.data_root_val
        ds_va = build_combined_dataset(root_val, args.T, args.n, False)
        val_loader = DataLoader(ds_va, args.batch, False)
        # evaluate(net, val_loader, 0, 0, steps, args, use_wandb=False)
        if args.out_dir:
            export_predictions(net, val_loader, steps, args)
    else:
        train(args)

if __name__=='__main__':
    main()
