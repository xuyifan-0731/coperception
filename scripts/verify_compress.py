# verify_compress.py
# 说明：
# - 在 Encoder 和 Decoder 之间验证压缩：统计“encoder隐藏状态(h,c)整段时间序列”的压缩率。
# - 支持三种模式：
#     1) lossless32  : 完全无损 float32 → (通道差分 + 空间左预测) + 时序Δ + 字节shuffle + zlib
#     2) fp16        : 先 FP16 量化（有损），再走上面同一套“无损”管线；统计 NMSE(%)
#     3) int8pc      : per-channel uint8 量化（有损），再做 时序Δ(模256) + 空间左预测(模256) + zlib；统计 NMSE(%)
# - 统一用“float32 全时序 h/c 大小”作为原始 baseline，用于对比压缩比。
# - 打印总原始大小、压缩后大小、平均压缩率，以及无损/有损的指标（lossless 通过率或 NMSE%）。

import os, glob, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import zlib

# ========================
# Args & Device
# ========================

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', nargs='+', required=True,
                   help='List of dataset root paths（与原脚本相同结构）')
    p.add_argument('--T', type=int, default=20, help='history length')
    p.add_argument('--n', type=int, default=10, help='total span; predict n-1 frames (仅占位，不训练)')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--H', type=int, default=256)
    p.add_argument('--W', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--print_each', action='store_true',
                   help='打印每个样本的压缩率与无损/误差结果')
    p.add_argument('--modes', type=str, default='lossless32,fp16,int8pc,int8pcsp,int8pc-adaptive,int4pc',
                   help='逗号分隔，可选：lossless32,fp16,int8pc')
    p.add_argument('--sparse_tau', type=int, default=1,
                   help='int8pcsp 稀疏阈值，|signed_delta| <= tau 置0（0=仅量化，无稀疏）')
    p.add_argument('--sparse_topk', type=float, default=0.0,
                   help='备用：若 >0 则优先按比例保留Top-K（0.1表示保留10%%非零），覆盖tau逻辑')
    return p.parse_args()

def pick_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================
# Utilities
# ========================

def sparse2bev(idx, shape):
    bev = np.zeros(shape, np.uint8)
    if idx.size == 0:
        return bev
    y, x = idx[:, 0], idx[:, 1]
    m = (y >= 0) & (y < shape[0]) & (x >= 0) & (x < shape[1])
    bev[y[m], x[m]] = 1
    return bev

# ========================
# Dataset
# ========================

class SeqDataset(Dataset):
    def __init__(self, root, T, n, H=256, W=256):
        assert n >= 2
        self.root, self.T, self.n = root, T, n
        self.H, self.W = H, W
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
        X = torch.from_numpy(np.stack(X)).unsqueeze(1).float()  # (T,1,H,W)
        return X, sc, hist_ids, tgt_ids

def build_combined_dataset(roots, T, n, H, W):
    datasets = [SeqDataset(root, T, n, H, W) for root in roots]
    class _Concat(Dataset):
        def __init__(self, ds_list):
            self.ds_list = ds_list
            self.cum = []
            s = 0
            for d in ds_list:
                self.cum.append((s, s+len(d), d))
                s += len(d)
            self.N = s
        def __len__(self): return self.N
        def __getitem__(self, i):
            for s,e,d in self.cum:
                if s <= i < e:
                    return d[i - s]
            raise IndexError
    return _Concat(datasets)

# ========================
# Model: ConvLSTM Encoder（返回全时序状态）
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
        self.c_hid = c_hid
    def forward(self, x):
        B, T, C, H, W = x.shape
        h = x.new_zeros((B, self.c_hid, H, W))
        c = x.new_zeros((B, self.c_hid, H, W))
        H_seq, C_seq = [], []
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
            H_seq.append(h)
            C_seq.append(c)
        return h, c, H_seq, C_seq

# ========================
# 可逆预处理（float 路线）
# ========================

def bytes_shuffle(arr: np.ndarray) -> bytes:
    assert arr.flags['C_CONTIGUOUS'], "arr 需为 C 连续内存"
    itemsize = arr.dtype.itemsize
    raw = memoryview(arr).cast('B')
    n = arr.size
    if n == 0: return b""
    tmp = np.frombuffer(raw, dtype=np.uint8).reshape(n, itemsize)
    shuffled = tmp.T.copy().reshape(-1)
    return shuffled.tobytes()

def bytes_unshuffle(buf: bytes, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    itemsize = np.dtype(dtype).itemsize
    n = int(np.prod(shape))
    if n == 0:
        return np.empty(shape, dtype=dtype)
    tmp = np.frombuffer(buf, dtype=np.uint8).reshape(itemsize, n)
    unshuffled = tmp.T.copy().reshape(n * itemsize)
    out = np.frombuffer(unshuffled, dtype=dtype).copy().reshape(shape)
    return out

def _left_filter(arr: np.ndarray) -> np.ndarray:
    # 沿最后一维做左预测：y[...,0]=x[...,0], y[...,1:]=x[...,1:]-x[...,:-1]
    out = np.empty_like(arr)
    out[..., :, 0] = arr[..., :, 0]
    out[..., :, 1:] = arr[..., :, 1:] - arr[..., :, :-1]
    return out

def _left_unfilter(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr)
    out[..., :, 0] = arr[..., :, 0]
    if arr.shape[-1] > 1:
        csum = out[..., :, 0].copy()
        for x in range(1, arr.shape[-1]):
            csum = csum + arr[..., :, x]
            out[..., :, x] = csum
    return out

def _channel_delta(arr: np.ndarray) -> np.ndarray:
    # arr: (2,C,H,W) 或 (C,H,W)
    if arr.ndim == 4:
        first = arr[:, 0:1]
        rest  = arr[:, 1:] - arr[:, :-1]
        return np.concatenate([first, rest], axis=1)
    elif arr.ndim == 3:
        first = arr[0:1]
        rest  = arr[1:] - arr[:-1]
        return np.concatenate([first, rest], axis=0)
    else:
        raise ValueError("shape not supported")

def _channel_undelta(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    if arr.ndim == 4:
        for i in range(1, out.shape[1]):
            out[:, i] = out[:, i] + out[:, i-1]
    else:
        for i in range(1, out.shape[0]):
            out[i] = out[i] + out[i-1]
    return out

def _prep_and_compress(arr: np.ndarray, dtype='float32') -> bytes:
    # arr -> (通道差分 + 左预测) -> bytes shuffle -> zlib
    arr = np.ascontiguousarray(arr.astype(dtype, copy=False))
    arr = _channel_delta(arr)
    arr = _left_filter(arr)
    shuf = bytes_shuffle(arr)
    return zlib.compress(shuf, level=9)

def _decompress_and_restore(buf: bytes, shape_after_stack: tuple, dtype='float32') -> np.ndarray:
    raw = zlib.decompress(buf)
    arr = bytes_unshuffle(raw, np.dtype(dtype), shape_after_stack)
    arr = _left_unfilter(arr)
    arr = _channel_undelta(arr)
    return arr

# ========================
# 时序 Δ（float 路线：关键帧 + 残差）
# ========================

def compress_hc_sequence_lossless(h_seq_t, c_seq_t, dtype='float32'):
    """
    h_seq_t/c_seq_t: 长度 T 的 list，每个元素 (C,H,W) float32 或 float16
    返回：keyframe payload + deltas payload 列表 + meta
    """
    T = len(h_seq_t)
    assert T == len(c_seq_t) and T >= 1
    C, H, W = h_seq_t[0].shape
    meta = {
        "T": T,
        "orig_shape": (2, C, H, W),
        "dtype": dtype,
    }

    # t=0 关键帧
    h0 = h_seq_t[0].detach().cpu().contiguous().numpy().astype(dtype, copy=False)
    c0 = c_seq_t[0].detach().cpu().contiguous().numpy().astype(dtype, copy=False)
    kf = np.stack([h0, c0], axis=0)  # (2,C,H,W)
    key_payload = _prep_and_compress(kf, dtype=dtype)

    # t>=1 残差
    delta_payloads = []
    prev_h = h0; prev_c = c0
    for t in range(1, T):
        ht = h_seq_t[t].detach().cpu().contiguous().numpy().astype(dtype, copy=False)
        ct = c_seq_t[t].detach().cpu().contiguous().numpy().astype(dtype, copy=False)
        dh = ht - prev_h
        dc = ct - prev_c
        d = np.stack([dh, dc], axis=0)  # (2,C,H,W)
        delta_payloads.append(_prep_and_compress(d, dtype=dtype))
        prev_h, prev_c = ht, ct

    return {"meta": meta, "keyframe": key_payload, "deltas": delta_payloads}

def decompress_hc_sequence_lossless(pkg):
    meta = pkg["meta"]
    T = meta["T"]
    shape = meta["orig_shape"]
    dtype = meta["dtype"]

    # 还原 t=0
    kf = _decompress_and_restore(pkg["keyframe"], shape_after_stack=shape, dtype=dtype)  # (2,C,H,W)
    h0 = kf[0]; c0 = kf[1]
    H_seq = [torch.from_numpy(h0.copy()).float()]
    C_seq = [torch.from_numpy(c0.copy()).float()]
    prev_h = h0; prev_c = c0

    # 依次还原 Δ
    for pay in pkg["deltas"]:
        d = _decompress_and_restore(pay, shape_after_stack=shape, dtype=dtype)  # (2,C,H,W)
        dh, dc = d[0], d[1]
        h = prev_h + dh
        c = prev_c + dc
        H_seq.append(torch.from_numpy(h.copy()).float())
        C_seq.append(torch.from_numpy(c.copy()).float())
        prev_h, prev_c = h, c

    return H_seq, C_seq

# ========================
# uint8 路线所需：模256 左预测/反预测 + 时序Δ/反Δ（全部可逆）
# ========================

def u8_left_filter(arr_u8: np.ndarray) -> np.ndarray:
    # arr_u8: uint8, shape (..., W)
    out = np.empty_like(arr_u8)
    out[..., 0] = arr_u8[..., 0]
    if arr_u8.shape[-1] > 1:
        diff = (arr_u8[..., 1:].astype(np.int16) - arr_u8[..., :-1].astype(np.int16)) & 0xFF
        out[..., 1:] = diff.astype(np.uint8)
    return out

def u8_left_unfilter(arr_u8: np.ndarray) -> np.ndarray:
    # 反变换：模256 累加
    tmp = arr_u8.astype(np.uint16)
    out = np.cumsum(tmp, axis=-1) & 0xFF
    return out.astype(np.uint8)

def u8_sub_mod(a_u8: np.ndarray, b_u8: np.ndarray) -> np.ndarray:
    return ((a_u8.astype(np.int16) - b_u8.astype(np.int16)) & 0xFF).astype(np.uint8)

def u8_add_mod(a_u8: np.ndarray, b_u8: np.ndarray) -> np.ndarray:
    return ((a_u8.astype(np.int16) + b_u8.astype(np.int16)) & 0xFF).astype(np.uint8)

# ========================
# per-channel uint8 量化 + 时序Δ(模256) + 空间左预测(模256)
# ========================

def _compute_pc_minmax(stack_TCHW: np.ndarray):
    # 输入 (T, C, H, W) float32，返回 per-channel min/max，shape (C,)
    mins = stack_TCHW.min(axis=(0,2,3))
    maxs = stack_TCHW.max(axis=(0,2,3))
    return mins.astype(np.float32), maxs.astype(np.float32)

def _pc_quantize_uint8(x: np.ndarray, ch_mins: np.ndarray, ch_maxs: np.ndarray) -> np.ndarray:
    # x: (C,H,W) float32, per-channel线性量化到 [0,255]
    C = x.shape[0]
    eps = 1e-8
    scale = (ch_maxs - ch_mins) / 255.0
    scale = np.where(scale < eps, 1.0, scale)
    z = (x - ch_mins.reshape(C,1,1)) / scale.reshape(C,1,1)
    z = np.round(z)
    z = np.clip(z, 0, 255).astype(np.uint8)
    return z

def _pc_dequantize_uint8(q: np.ndarray, ch_mins: np.ndarray, ch_maxs: np.ndarray) -> np.ndarray:
    C = q.shape[0]
    scale = (ch_maxs - ch_mins) / 255.0
    scale = np.where(scale < 1e-8, 1.0, scale)
    x = q.astype(np.float32) * scale.reshape(C,1,1) + ch_mins.reshape(C,1,1)
    return x.astype(np.float32)

def compress_hc_sequence_int8pc(h_seq_t, c_seq_t):
    """
    per-channel uint8 量化 + (关键帧 + 时序Δ mod256) + 空间左预测 mod256 + zlib
    返回：payloads + meta（包含 per-channel min/max）
    """
    T = len(h_seq_t)
    assert T == len(c_seq_t) and T >= 1
    C, H, W = h_seq_t[0].shape

    # 先把整段叠起来，算 per-channel min/max（h 与 c 分开）
    H_stack = np.stack([h.detach().cpu().numpy().astype(np.float32, copy=False) for h in h_seq_t], axis=0)  # (T,C,H,W)
    C_stack = np.stack([c.detach().cpu().numpy().astype(np.float32, copy=False) for c in c_seq_t], axis=0)  # (T,C,H,W)
    h_mins, h_maxs = _compute_pc_minmax(H_stack)
    c_mins, c_maxs = _compute_pc_minmax(C_stack)

    # t=0
    qh0 = _pc_quantize_uint8(H_stack[0], h_mins, h_maxs)
    qc0 = _pc_quantize_uint8(C_stack[0], c_mins, c_maxs)
    arr0 = np.stack([qh0, qc0], axis=0)  # (2,C,H,W) uint8
    key_payload = zlib.compress(u8_left_filter(arr0), level=9)

    # t>=1
    delta_payloads = []
    prev = arr0
    for t in range(1, T):
        qht = _pc_quantize_uint8(H_stack[t], h_mins, h_maxs)
        qct = _pc_quantize_uint8(C_stack[t], c_mins, c_maxs)
        arrt = np.stack([qht, qct], axis=0)
        d = u8_sub_mod(arrt, prev)             # 时序Δ（模256）
        delta_payloads.append(zlib.compress(u8_left_filter(d), level=9))
        prev = arrt

    meta = {
        "T": T,
        "shape_u8": (2, C, H, W),
        "h_mins": h_mins, "h_maxs": h_maxs,
        "c_mins": c_mins, "c_maxs": c_maxs,
    }
    return {"meta": meta, "keyframe": key_payload, "deltas": delta_payloads}

def decompress_hc_sequence_int8pc(pkg):
    meta = pkg["meta"]
    T = meta["T"]
    shape = meta["shape_u8"]
    h_mins, h_maxs = meta["h_mins"], meta["h_maxs"]
    c_mins, c_maxs = meta["c_mins"], meta["c_maxs"]

    # 还原 t=0 的量化 uint8
    arr0 = u8_left_unfilter(np.frombuffer(zlib.decompress(pkg["keyframe"]), dtype=np.uint8).reshape(shape))
    H_rec = []
    C_rec = []
    # 反量化到 float32
    h0 = _pc_dequantize_uint8(arr0[0], h_mins, h_maxs)
    c0 = _pc_dequantize_uint8(arr0[1], c_mins, c_maxs)
    H_rec.append(torch.from_numpy(h0.copy()))
    C_rec.append(torch.from_numpy(c0.copy()))

    prev = arr0
    for pay in pkg["deltas"]:
        d = u8_left_unfilter(np.frombuffer(zlib.decompress(pay), dtype=np.uint8).reshape(shape))
        arrt = u8_add_mod(prev, d)  # 累加（模256）
        ht = _pc_dequantize_uint8(arrt[0], h_mins, h_maxs)
        ct = _pc_dequantize_uint8(arrt[1], c_mins, c_maxs)
        H_rec.append(torch.from_numpy(ht.copy()))
        C_rec.append(torch.from_numpy(ct.copy()))
        prev = arrt

    # 转 float32 tensor
    H_rec = [x.float() for x in H_rec]
    C_rec = [x.float() for x in C_rec]
    return H_rec, C_rec

# ========================
# int8pcsp 路线所需：per-channel uint8 量化 + 时序Δ(模256) + 空间左预测(模256) + 稀疏
# ========================

# ---------- 工具：u8 <-> 有符号残差 ----------
def u8_to_signed(d_u8: np.ndarray) -> np.ndarray:
    # uint8 [0..255] -> int8 [-128..127]（按字节解释）
    return ((d_u8.astype(np.int16) + 128) & 0xFF).astype(np.uint8).view(np.int8)

def signed_to_u8(s_i8: np.ndarray) -> np.ndarray:
    # int8 -> uint8
    return (s_i8.astype(np.int16) - (-128)) & 0xFF

def flat_sparse_pack(idx_i32: np.ndarray, val_i8: np.ndarray) -> bytes:
    # 组包：先拼长度，再拼两段
    head = np.array([idx_i32.size, val_i8.size], dtype=np.int32).tobytes()
    body = idx_i32.tobytes() + val_i8.tobytes()
    return zlib.compress(head + body, level=9)

def flat_sparse_unpack(buf: bytes) -> tuple[np.ndarray, np.ndarray]:
    raw = zlib.decompress(buf)
    n_idx, n_val = np.frombuffer(raw[:8], dtype=np.int32)
    payload = raw[8:]
    idx_bytes = n_idx * 4
    idx = np.frombuffer(payload[:idx_bytes], dtype=np.int32)
    val = np.frombuffer(payload[idx_bytes:idx_bytes+n_val], dtype=np.int8)
    return idx.copy(), val.copy()

def compress_hc_sequence_int8pcsp(h_seq_t, c_seq_t, tau=1, topk_ratio=0.0):
    """
    per-channel uint8 量化 + (时序Δ, 有符号) + 稀疏化 + zlib
    t=0 发整帧（zlib）；t>=1 发稀疏残差（索引+值，zlib）。
    """
    T = len(h_seq_t)
    assert T == len(c_seq_t) and T >= 1
    C, H, W = h_seq_t[0].shape

    # 叠全时序，计算 per-channel min/max
    H_stack = np.stack([h.detach().cpu().numpy().astype(np.float32, copy=False) for h in h_seq_t], axis=0)
    C_stack = np.stack([c.detach().cpu().numpy().astype(np.float32, copy=False) for c in c_seq_t], axis=0)
    h_mins, h_maxs = _compute_pc_minmax(H_stack)
    c_mins, c_maxs = _compute_pc_minmax(C_stack)

    # t=0：量化并打包（整帧）
    qh0 = _pc_quantize_uint8(H_stack[0], h_mins, h_maxs)
    qc0 = _pc_quantize_uint8(C_stack[0], c_mins, c_maxs)
    arr0 = np.stack([qh0, qc0], axis=0)  # (2,C,H,W) uint8
    key_payload = zlib.compress(arr0, level=9)

    # t>=1：量化 -> 与上帧做无符号差，再转有符号残差 -> 稀疏化 -> 打包
    delta_payloads = []
    prev = arr0
    N = np.prod(arr0.shape, dtype=np.int64)  # 展平长度
    for t in range(1, T):
        qht = _pc_quantize_uint8(H_stack[t], h_mins, h_maxs)
        qct = _pc_quantize_uint8(C_stack[t], c_mins, c_maxs)
        curr = np.stack([qht, qct], axis=0).astype(np.uint8, copy=False)

        # 无符号差（mod 256）
        d_u8 = u8_sub_mod(curr, prev)              # uint8
        # 映射到有符号域 [-128..127]
        d_i8 = u8_to_signed(d_u8).view(np.int8)

        # 稀疏化：阈值或Top-K（二选一）
        flat = d_i8.reshape(-1)
        if topk_ratio > 0.0:
            k = max(1, int(topk_ratio * flat.size))
            # Top-K by |delta|
            if k < flat.size:
                sel = np.argpartition(np.abs(flat), -k)[-k:]
            else:
                sel = np.arange(flat.size, dtype=np.int32)
            idx = np.sort(sel.astype(np.int32))
            val = flat[idx]
        else:
            # 阈值 tau：|Δ|<=tau 置0，只保留显著残差
            mask = (flat > tau) | (flat < -tau)
            idx = np.flatnonzero(mask).astype(np.int32)
            val = flat[idx]

        # 打包本帧稀疏
        payload = flat_sparse_pack(idx, val.astype(np.int8))
        delta_payloads.append(payload)
        prev = curr

    meta = {
        "T": T, "shape_u8": (2, C, H, W),
        "h_mins": h_mins, "h_maxs": h_maxs,
        "c_mins": c_mins, "c_maxs": c_maxs,
        "tau": int(tau), "topk_ratio": float(topk_ratio),
    }
    return {"meta": meta, "keyframe": key_payload, "deltas": delta_payloads}

def decompress_hc_sequence_int8pcsp(pkg):
    meta = pkg["meta"]
    T = meta["T"]
    shape = meta["shape_u8"]
    h_mins, h_maxs = meta["h_mins"], meta["h_maxs"]
    c_mins, c_maxs = meta["c_mins"], meta["c_maxs"]

    # 还原 t=0
    arr0 = np.frombuffer(zlib.decompress(pkg["keyframe"]), dtype=np.uint8).reshape(shape)
    H_rec, C_rec = [], []

    h0 = _pc_dequantize_uint8(arr0[0], h_mins, h_maxs)
    c0 = _pc_dequantize_uint8(arr0[1], c_mins, c_maxs)
    H_rec.append(torch.from_numpy(h0.copy()))
    C_rec.append(torch.from_numpy(c0.copy()))
    prev = arr0

    # 逐帧还原稀疏残差
    N = np.prod(shape, dtype=np.int64)
    for pay in pkg["deltas"]:
        idx, val = flat_sparse_unpack(pay)  # idx:int32, val:int8
        d_i8 = np.zeros(N, dtype=np.int8)
        if idx.size > 0:
            d_i8[idx] = val
        d_i8 = d_i8.reshape(shape)
        # 回到 uint8 域
        d_u8 = signed_to_u8(d_i8)
        curr = u8_add_mod(prev, d_u8)

        h = _pc_dequantize_uint8(curr[0], h_mins, h_maxs)
        c = _pc_dequantize_uint8(curr[1], c_mins, c_maxs)
        H_rec.append(torch.from_numpy(h.copy()))
        C_rec.append(torch.from_numpy(c.copy()))
        prev = curr

    H_rec = [x.float() for x in H_rec]
    C_rec = [x.float() for x in C_rec]
    return H_rec, C_rec

# ========================
# int8pc_adaptive 主要策略：1动态量化： 使用 自适应量化，通过根据数据分布动态调整量化位数，避免过度压缩。2自适应稀疏化： 根据数据的能量分布，动态调整是否使用稀疏化，避免稀疏化带来的元数据开销。3结合 int8pc 和 int8pcsp 的优点： 在数据稀疏的区域使用 int8pcsp，而在较为密集的区域使用 int8pc，从而达到最佳压缩效果。
# ========================


def compress_hc_sequence_int8pc_adaptive(h_seq_t, c_seq_t, tau=1, topk_ratio=0.02):
    """
    自适应模式：基于时序和空间差分动态调整量化精度和稀疏化策略。
    """
    T = len(h_seq_t)
    assert T == len(c_seq_t) and T >= 1
    C, H, W = h_seq_t[0].shape

    # 叠全时序，计算 per-channel min/max
    H_stack = np.stack([h.detach().cpu().numpy().astype(np.float32, copy=False) for h in h_seq_t], axis=0)
    C_stack = np.stack([c.detach().cpu().numpy().astype(np.float32, copy=False) for c in c_seq_t], axis=0)
    h_mins, h_maxs = _compute_pc_minmax(H_stack)
    c_mins, c_maxs = _compute_pc_minmax(C_stack)

    # t=0：量化并打包（整帧）
    qh0 = _pc_quantize_uint8(H_stack[0], h_mins, h_maxs)
    qc0 = _pc_quantize_uint8(C_stack[0], c_mins, c_maxs)
    arr0 = np.stack([qh0, qc0], axis=0)  # (2,C,H,W) uint8
    key_payload = zlib.compress(arr0, level=9)

    # t>=1：量化 -> 与上帧做无符号差，再转有符号残差 -> 稀疏化 -> 打包
    delta_payloads = []
    prev = arr0
    N = np.prod(arr0.shape, dtype=np.int64)  # 展平长度
    for t in range(1, T):
        qht = _pc_quantize_uint8(H_stack[t], h_mins, h_maxs)
        qct = _pc_quantize_uint8(C_stack[t], c_mins, c_maxs)
        curr = np.stack([qht, qct], axis=0).astype(np.uint8, copy=False)

        # 无符号差（mod 256）
        d_u8 = u8_sub_mod(curr, prev)              # uint8
        # 映射到有符号域 [-128..127]
        d_i8 = u8_to_signed(d_u8).view(np.int8)

        # 动态稀疏化策略：根据能量选择是否稀疏
        flat = d_i8.reshape(-1)
        energy = np.sum(flat**2)
        
        # 稀疏化条件判断
        if energy > 10000:  # 动态判断是否需要稀疏化
            k = max(1, int(topk_ratio * flat.size))
            # Top-K by |delta|
            if k < flat.size:
                sel = np.argpartition(np.abs(flat), -k)[-k:]
            else:
                sel = np.arange(flat.size, dtype=np.int32)
            idx = np.sort(sel.astype(np.int32))
            val = flat[idx]
            payload = flat_sparse_pack(idx, val.astype(np.int8))
        else:
            # 无需稀疏化，直接量化
            idx = np.flatnonzero(flat > tau).astype(np.int32)
            val = flat[idx]
            payload = flat_sparse_pack(idx, val.astype(np.int8))

        delta_payloads.append(payload)
        prev = curr

    meta = {
        "T": T, "shape_u8": (2, C, H, W),
        "h_mins": h_mins, "h_maxs": h_maxs,
        "c_mins": c_mins, "c_maxs": c_maxs,
        "tau": int(tau), "topk_ratio": float(topk_ratio),
    }
    return {"meta": meta, "keyframe": key_payload, "deltas": delta_payloads}


def decompress_hc_sequence_int8pc_adaptive(pkg):
    meta = pkg["meta"]
    T = meta["T"]
    shape = meta["shape_u8"]
    h_mins, h_maxs = meta["h_mins"], meta["h_maxs"]
    c_mins, c_maxs = meta["c_mins"], meta["c_maxs"]

    # 还原 t=0
    arr0 = np.frombuffer(zlib.decompress(pkg["keyframe"]), dtype=np.uint8).reshape(shape)
    H_rec, C_rec = [], []

    h0 = _pc_dequantize_uint8(arr0[0], h_mins, h_maxs)
    c0 = _pc_dequantize_uint8(arr0[1], c_mins, c_maxs)
    H_rec.append(torch.from_numpy(h0.copy()))
    C_rec.append(torch.from_numpy(c0.copy()))
    prev = arr0

    # 逐帧还原稀疏残差
    N = np.prod(shape, dtype=np.int64)
    for pay in pkg["deltas"]:
        idx, val = flat_sparse_unpack(pay)  # idx:int32, val:int8
        d_i8 = np.zeros(N, dtype=np.int8)
        if idx.size > 0:
            d_i8[idx] = val
        d_i8 = d_i8.reshape(shape)
        # 回到 uint8 域
        d_u8 = signed_to_u8(d_i8)
        curr = u8_add_mod(prev, d_u8)

        h = _pc_dequantize_uint8(curr[0], h_mins, h_maxs)
        c = _pc_dequantize_uint8(curr[1], c_mins, c_maxs)
        H_rec.append(torch.from_numpy(h.copy()))
        C_rec.append(torch.from_numpy(c.copy()))
        prev = curr

    H_rec = [x.float() for x in H_rec]
    C_rec = [x.float() for x in C_rec]
    return H_rec, C_rec

# ========================
# int4
# ========================
def _pc_quantize_int4(x: np.ndarray, ch_mins: np.ndarray, ch_maxs: np.ndarray) -> np.ndarray:
    """
    使用 int4 量化方法，返回量化后的 uint4 数值
    x: (C,H,W) float32
    ch_mins: 每个通道的最小值
    ch_maxs: 每个通道的最大值
    """
    C = x.shape[0]
    eps = 1e-8
    scale = (ch_maxs - ch_mins) / 15.0  # 0-15 对应 int4 的范围
    scale = np.where(scale < eps, 1.0, scale)
    z = (x - ch_mins.reshape(C,1,1)) / scale.reshape(C,1,1)
    z = np.round(z)
    z = np.clip(z, 0, 15).astype(np.uint8)  # uint4，但需要转成 uint8 存储
    return z

def _pc_dequantize_int4(q: np.ndarray, ch_mins: np.ndarray, ch_maxs: np.ndarray) -> np.ndarray:
    """
    反量化 int4 数据，恢复为 float32
    q: (C,H,W) uint4
    ch_mins: 每个通道的最小值
    ch_maxs: 每个通道的最大值
    """
    C = q.shape[0]
    scale = (ch_maxs - ch_mins) / 15.0  # int4
    scale = np.where(scale < 1e-8, 1.0, scale)
    x = q.astype(np.float32) * scale.reshape(C,1,1) + ch_mins.reshape(C,1,1)
    return x

def compress_hc_sequence_int4pc(h_seq_t, c_seq_t):
    """
    使用 int4 量化（代替原 int8pc 的量化），然后压缩。
    """
    T = len(h_seq_t)
    assert T == len(c_seq_t) and T >= 1
    C, H, W = h_seq_t[0].shape

    # 叠全时序，计算 per-channel min/max（h 与 c 分开）
    H_stack = np.stack([h.detach().cpu().numpy().astype(np.float32, copy=False) for h in h_seq_t], axis=0)  # (T,C,H,W)
    C_stack = np.stack([c.detach().cpu().numpy().astype(np.float32, copy=False) for c in c_seq_t], axis=0)  # (T,C,H,W)
    h_mins, h_maxs = _compute_pc_minmax(H_stack)
    c_mins, c_maxs = _compute_pc_minmax(C_stack)

    # t=0：量化并打包（整帧）
    qh0 = _pc_quantize_int4(H_stack[0], h_mins, h_maxs)
    qc0 = _pc_quantize_int4(C_stack[0], c_mins, c_maxs)
    arr0 = np.stack([qh0, qc0], axis=0)  # (2,C,H,W) uint4
    key_payload = zlib.compress(arr0, level=9)

    # t>=1：量化 -> 与上帧做无符号差，再转有符号残差 -> 稀疏化 -> 打包
    delta_payloads = []
    prev = arr0
    for t in range(1, T):
        qht = _pc_quantize_int4(H_stack[t], h_mins, h_maxs)
        qct = _pc_quantize_int4(C_stack[t], c_mins, c_maxs)
        curr = np.stack([qht, qct], axis=0).astype(np.uint8, copy=False)

        # 无符号差（mod 16，因为是 int4）
        d_u8 = u8_sub_mod(curr, prev)              # uint8
        # 映射到有符号域 [-8..7]
        d_i8 = u8_to_signed(d_u8).view(np.int8)

        # 稀疏化：Top-K（二选一）
        flat = d_i8.reshape(-1)
        k = max(1, int(0.1 * flat.size))  # 默认 Top 10%
        if k < flat.size:
            sel = np.argpartition(np.abs(flat), -k)[-k:]
        else:
            sel = np.arange(flat.size, dtype=np.int32)
        idx = np.sort(sel.astype(np.int32))
        val = flat[idx]

        # 打包本帧稀疏
        payload = flat_sparse_pack(idx, val.astype(np.int8))
        delta_payloads.append(payload)
        prev = curr

    meta = {
        "T": T, "shape_u8": (2, C, H, W),
        "h_mins": h_mins, "h_maxs": h_maxs,
        "c_mins": c_mins, "c_maxs": c_maxs,
    }
    return {"meta": meta, "keyframe": key_payload, "deltas": delta_payloads}

def decompress_hc_sequence_int4pc(pkg):
    meta = pkg["meta"]
    T = meta["T"]
    shape = meta["shape_u8"]
    h_mins, h_maxs = meta["h_mins"], meta["h_maxs"]
    c_mins, c_maxs = meta["c_mins"], meta["c_maxs"]

    # 还原 t=0
    arr0 = np.frombuffer(zlib.decompress(pkg["keyframe"]), dtype=np.uint8).reshape(shape)
    H_rec, C_rec = [], []

    h0 = _pc_dequantize_int4(arr0[0], h_mins, h_maxs)
    c0 = _pc_dequantize_int4(arr0[1], c_mins, c_maxs)
    H_rec.append(torch.from_numpy(h0.copy()))
    C_rec.append(torch.from_numpy(c0.copy()))
    prev = arr0

    # 逐帧还原稀疏残差
    for pay in pkg["deltas"]:
        idx, val = flat_sparse_unpack(pay)  # idx:int32, val:int8
        d_i8 = np.zeros(np.prod(shape), dtype=np.int8)
        if idx.size > 0:
            d_i8[idx] = val
        d_i8 = d_i8.reshape(shape)
        # 回到 uint8 域
        d_u8 = signed_to_u8(d_i8)
        curr = u8_add_mod(prev, d_u8)

        h = _pc_dequantize_int4(curr[0], h_mins, h_maxs)
        c = _pc_dequantize_int4(curr[1], c_mins, c_maxs)
        H_rec.append(torch.from_numpy(h.copy()))
        C_rec.append(torch.from_numpy(c.copy()))
        prev = curr

    H_rec = [x.float() for x in H_rec]
    C_rec = [x.float() for x in C_rec]
    return H_rec, C_rec



# ========================
# 指标：NMSE（误差占比）
# ========================

def nmse_percent(orig_list, rec_list):
    # 在全时序、h/c 全部元素上统计 NMSE(%)
    num = 0.0
    den = 0.0
    for o, r in zip(orig_list, rec_list):
        o_np = o.detach().cpu().numpy().astype(np.float64, copy=False)
        r_np = r.detach().cpu().numpy().astype(np.float64, copy=False)
        diff = o_np - r_np
        num += np.sum(diff * diff)
        den += np.sum(o_np * o_np) + 1e-12
    return (num / den) * 100.0, num, den

# ========================
# 主验证循环
# ========================

def verify_loop(encoder, loader, device, modes, print_each=False):
    encoder.eval()
    modes = [m.strip().lower() for m in modes.split(',') if m.strip()]

    # 统计字典
    stats = {
        m: {
            'orig_bytes': 0,
            'comp_bytes': 0,
            'samples': 0,
            'lossless_ok': 0,   # 仅对 lossless32 有意义
            'nmse_num': 0.0,    # 对有损模式有效
            'nmse_den': 0.0,
            'meta_bytes': 0,    # int8pc 元信息开销
        } for m in modes
    }

    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc="Verifying compression (encoder states)")
        for X, *_ in pbar:
            X = X.to(device)  # (B,T,1,H,W)
            _, _, H_seq_all, C_seq_all = encoder(X)   # list 长度 T，每个 (B,C,H,W)
            T = len(H_seq_all)
            B = X.size(0)

            for b in range(B):
                # 抽出单个样本的整段 h/c（list[Tensor], each (C,H,W) float32）
                h_seq_t = [H_seq_all[t][b].float().cpu() for t in range(T)]
                c_seq_t = [C_seq_all[t][b].float().cpu() for t in range(T)]

                C_, H_, W_ = h_seq_t[0].shape
                # 原始 baseline 按 float32 全时序、h/c 计算
                orig_bytes = T * ((C_*H_*W_*4) * 2)

                # ====== 模式：lossless32 ======
                if 'lossless32' in stats:
                    pkg = compress_hc_sequence_lossless(h_seq_t, c_seq_t, dtype='float32')
                    comp_bytes = len(pkg["keyframe"]) + sum(len(p) for p in pkg["deltas"])
                    H_rec, C_rec = decompress_hc_sequence_lossless(pkg)
                    # 验真
                    ok = True
                    for t in range(T):
                        ok = ok and torch.equal(h_seq_t[t], H_rec[t]) and torch.equal(c_seq_t[t], C_rec[t])
                    stats['lossless32']['orig_bytes'] += orig_bytes
                    stats['lossless32']['comp_bytes'] += comp_bytes
                    stats['lossless32']['samples']    += 1
                    if ok: stats['lossless32']['lossless_ok'] += 1
                    if print_each:
                        ratio = comp_bytes / orig_bytes
                        print(f"[lossless32] raw={orig_bytes/1024/1024:.3f}MiB cmp={comp_bytes/1024/1024:.3f}MiB "
                              f"ratio={ratio:.4f} lossless={ok}")

                # ====== 模式：fp16（先量化再无损）======
                if 'fp16' in stats:
                    # 将列表转成 fp16 再压
                    h16 = [h.half() for h in h_seq_t]
                    c16 = [c.half() for c in c_seq_t]
                    pkg = compress_hc_sequence_lossless(h16, c16, dtype='float16')
                    comp_bytes = len(pkg["keyframe"]) + sum(len(p) for p in pkg["deltas"])
                    H_rec, C_rec = decompress_hc_sequence_lossless(pkg)  # 返回 float32
                    # 统计 NMSE(%)
                    nmse_h, num_h, den_h = nmse_percent(h_seq_t, H_rec)
                    nmse_c, num_c, den_c = nmse_percent(c_seq_t, C_rec)
                    stats['fp16']['orig_bytes'] += orig_bytes
                    stats['fp16']['comp_bytes'] += comp_bytes
                    stats['fp16']['samples']    += 1
                    stats['fp16']['nmse_num']   += (num_h + num_c)
                    stats['fp16']['nmse_den']   += (den_h + den_c)
                    if print_each:
                        ratio = comp_bytes / orig_bytes
                        nmse_tot = (num_h + num_c) / (den_h + den_c + 1e-12) * 100.0
                        print(f"[fp16]      raw={orig_bytes/1024/1024:.3f}MiB cmp={comp_bytes/1024/1024:.3f}MiB "
                              f"ratio={ratio:.4f} NMSE={nmse_tot:.4f}%")

                # ====== 模式：int8pc（per-channel uint8 + 可逆管线）======
                if 'int8pc' in stats:
                    pkg = compress_hc_sequence_int8pc(h_seq_t, c_seq_t)
                    # 元信息开销（min/max）
                    meta_bytes = (pkg["meta"]["h_mins"].nbytes + pkg["meta"]["h_maxs"].nbytes +
                                  pkg["meta"]["c_mins"].nbytes + pkg["meta"]["c_maxs"].nbytes)
                    comp_bytes = (len(pkg["keyframe"]) + sum(len(p) for p in pkg["deltas"]) + meta_bytes)
                    H_rec, C_rec = decompress_hc_sequence_int8pc(pkg)
                    nmse_h, num_h, den_h = nmse_percent(h_seq_t, H_rec)
                    nmse_c, num_c, den_c = nmse_percent(c_seq_t, C_rec)
                    stats['int8pc']['orig_bytes'] += orig_bytes
                    stats['int8pc']['comp_bytes'] += comp_bytes
                    stats['int8pc']['samples']    += 1
                    stats['int8pc']['nmse_num']   += (num_h + num_c)
                    stats['int8pc']['nmse_den']   += (den_h + den_c)
                    stats['int8pc']['meta_bytes'] += meta_bytes
                    if print_each:
                        ratio = comp_bytes / orig_bytes
                        nmse_tot = (num_h + num_c) / (den_h + den_c + 1e-12) * 100.0
                        print(f"[int8pc]    raw={orig_bytes/1024/1024:.3f}MiB cmp={comp_bytes/1024/1024:.3f}MiB "
                              f"ratio={ratio:.4f} NMSE={nmse_tot:.4f}% (meta {meta_bytes}B)")

                # ====== 模式：int8pc（per-channel uint8 + 可逆管线）======
                if 'int4pc' in stats:
                    pkg = compress_hc_sequence_int4pc(h_seq_t, c_seq_t)
                    # 元信息开销（min/max）
                    meta_bytes = (pkg["meta"]["h_mins"].nbytes + pkg["meta"]["h_maxs"].nbytes +
                                  pkg["meta"]["c_mins"].nbytes + pkg["meta"]["c_maxs"].nbytes)
                    comp_bytes = (len(pkg["keyframe"]) + sum(len(p) for p in pkg["deltas"]) + meta_bytes)
                    H_rec, C_rec = decompress_hc_sequence_int4pc(pkg)
                    nmse_h, num_h, den_h = nmse_percent(h_seq_t, H_rec)
                    nmse_c, num_c, den_c = nmse_percent(c_seq_t, C_rec)
                    stats['int4pc']['orig_bytes'] += orig_bytes
                    stats['int4pc']['comp_bytes'] += comp_bytes
                    stats['int4pc']['samples']    += 1
                    stats['int4pc']['nmse_num']   += (num_h + num_c)
                    stats['int4pc']['nmse_den']   += (den_h + den_c)
                    stats['int4pc']['meta_bytes'] += meta_bytes
                    if print_each:
                        ratio = comp_bytes / orig_bytes
                        nmse_tot = (num_h + num_c) / (den_h + den_c + 1e-12) * 100.0
                        print(f"[int4pc]    raw={orig_bytes/1024/1024:.3f}MiB cmp={comp_bytes/1024/1024:.3f}MiB "
                              f"ratio={ratio:.4f} NMSE={nmse_tot:.4f}% (meta {meta_bytes}B)")

                # ====== 模式：int8pcsp（INT8 + 稀疏）======
                if 'int8pcsp' in stats:
                    pkg = compress_hc_sequence_int8pcsp(
                        h_seq_t, c_seq_t,
                        tau=args.sparse_tau if 'args' in globals() else 1,
                        topk_ratio=args.sparse_topk if 'args' in globals() else 0.0
                    )
                    meta_bytes = (pkg["meta"]["h_mins"].nbytes + pkg["meta"]["h_maxs"].nbytes +
                                  pkg["meta"]["c_mins"].nbytes + pkg["meta"]["c_maxs"].nbytes)
                    comp_bytes = len(pkg["keyframe"]) + sum(len(p) for p in pkg["deltas"]) + meta_bytes
                    H_rec, C_rec = decompress_hc_sequence_int8pcsp(pkg)
                    nmse_h, num_h, den_h = nmse_percent(h_seq_t, H_rec)
                    nmse_c, num_c, den_c = nmse_percent(c_seq_t, C_rec)

                    stats['int8pcsp']['orig_bytes'] += orig_bytes
                    stats['int8pcsp']['comp_bytes'] += comp_bytes
                    stats['int8pcsp']['samples']    += 1
                    stats['int8pcsp']['nmse_num']   += (num_h + num_c)
                    stats['int8pcsp']['nmse_den']   += (den_h + den_c)
                    stats['int8pcsp']['meta_bytes'] += meta_bytes

                    if print_each:
                        ratio = comp_bytes / orig_bytes
                        nmse_tot = (num_h + num_c) / (den_h + den_c + 1e-12) * 100.0
                        print(f"[int8pcsp]  raw={orig_bytes/1024/1024:.3f}MiB "
                              f"cmp={comp_bytes/1024/1024:.3f}MiB ratio={ratio:.4f} "
                              f"NMSE={nmse_tot:.4f}% (meta {meta_bytes}B)")

                # ====== 模式：int8pc_adaptive（自适应量化 + 自适应稀疏化）======
                # ====== 模式：int8pc-adaptive ======
                if 'int8pc-adaptive' in stats:
                    pkg = compress_hc_sequence_int8pc_adaptive(h_seq_t, c_seq_t)
                    meta_bytes = (pkg["meta"]["h_mins"].nbytes + pkg["meta"]["h_maxs"].nbytes +
                                  pkg["meta"]["c_mins"].nbytes + pkg["meta"]["c_maxs"].nbytes)
                    comp_bytes = len(pkg["keyframe"]) + sum(len(p) for p in pkg["deltas"]) + meta_bytes
                    H_rec, C_rec = decompress_hc_sequence_int8pc_adaptive(pkg)
                    nmse_h, num_h, den_h = nmse_percent(h_seq_t, H_rec)
                    nmse_c, num_c, den_c = nmse_percent(c_seq_t, C_rec)

                    stats['int8pc-adaptive']['orig_bytes'] += orig_bytes
                    stats['int8pc-adaptive']['comp_bytes'] += comp_bytes
                    stats['int8pc-adaptive']['samples']    += 1
                    stats['int8pc-adaptive']['nmse_num']   += (num_h + num_c)
                    stats['int8pc-adaptive']['nmse_den']   += (den_h + den_c)
                    stats['int8pc-adaptive']['meta_bytes'] += meta_bytes

                    if print_each:
                        ratio = comp_bytes / orig_bytes
                        nmse_tot = (num_h + num_c) / (den_h + den_c + 1e-12) * 100.0
                        print(f"[int8pc-adaptive] raw={orig_bytes/1024/1024:.3f}MiB "
                              f"cmp={comp_bytes/1024/1024:.3f}MiB ratio={ratio:.4f} "
                              f"NMSE={nmse_tot:.4f}% (meta {meta_bytes}B)")
            
            # 进度条显示“当前所有模式”的总体均值（以第一个模式为主显示）
            show_mode = modes[0]
            cur_ratio = (stats[show_mode]['comp_bytes'] /
                         max(1, stats[show_mode]['orig_bytes']))
            if show_mode == 'lossless32':
                ok_rate = stats[show_mode]['lossless_ok'] / max(1, stats[show_mode]['samples'])
                pbar.set_postfix(avg_ratio=f"{cur_ratio:.4f}", lossless_rate=f"{ok_rate:.4f}")
            else:
                nmse = (stats[show_mode]['nmse_num'] /
                        max(1e-12, stats[show_mode]['nmse_den'])) * 100.0
                pbar.set_postfix(avg_ratio=f"{cur_ratio:.4f}", nmse_percent=f"{nmse:.4f}")

    # ====== 汇总打印 ======
    print("\n======== Summary (baseline = float32 full sequence of h,c) ========")
    for m in modes:
        s = stats[m]
        avg_ratio = s['comp_bytes'] / max(1, s['orig_bytes'])
        print(f"[{m}] samples={s['samples']} | "
              f"total_raw={s['orig_bytes']/1024/1024:.3f} MiB | "
              f"total_cmp={s['comp_bytes']/1024/1024:.3f} MiB | "
              f"avg_ratio={avg_ratio:.6f}")
        if m == 'lossless32':
            ok_rate = s['lossless_ok'] / max(1, s['samples'])
            print(f"      lossless_pass_rate = {ok_rate*100:.2f}%")
        elif s['nmse_den'] > 0:
            nmse = (s['nmse_num'] / s['nmse_den']) * 100.0
            add = f" (meta_overhead={s['meta_bytes']} B)" if m == 'int8pc' else ""
            print(f"      NMSE = {nmse:.6f}%{add}")
        elif s['nmse_den'] > 0:
            nmse = (s['nmse_num'] / s['nmse_den']) * 100.0
            add = f" (meta_overhead={s['meta_bytes']} B)" if m in ('int8pc', 'int8pcsp') else ""
            print(f"      NMSE = {nmse:.6f}%{add}")

# ========================
# Main
# ========================

def main():
    args = get_args()
    device = pick_device()
    print(f'Using device: {device}')

    # Data
    ds = build_combined_dataset(args.data_root, args.T, args.n, args.H, args.W)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    # Encoder
    encoder = EncoderConvLSTM(1, 32).to(device)
    encoder.eval()

    verify_loop(encoder, dl, device, modes=args.modes, print_each=args.print_each)

if __name__ == '__main__':
    main()