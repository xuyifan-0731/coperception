#!/usr/bin/env python3
"""Train a DAIR-V2X box-motion compensator for detector-side Ours runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
DAIR_V2X = ROOT / "external" / "DAIR-V2X-isolated" / "gpu0" / "v2x"
for path in [
    DAIR_V2X,
    DAIR_V2X.parent,
    DAIR_V2X / "dataset",
    DAIR_V2X / "dataset" / "dataset_utils",
    DAIR_V2X / "models",
    DAIR_V2X / "models" / "model_utils",
    DAIR_V2X / "v2x_utils",
]:
    sys.path.insert(0, str(path))

from v2x_utils.transformation_utils import Coord_transformation  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dair-root", type=Path, default=ROOT / "datasets/DAIR-V2X/cooperative-vehicle-infrastructure")
    p.add_argument(
        "--split-data-path",
        type=Path,
        default=ROOT / "external/DAIR-V2X/data/split_datas/cooperative-split-data.json",
    )
    p.add_argument("--out", type=Path, default=ROOT / "results/260615_fill/artifacts/dair_ours_motion_compensator.npz")
    p.add_argument("--horizons", default="1,2,3,4,5,6,7,8,9")
    p.add_argument("--max-match-dist", type=float, default=8.0)
    p.add_argument("--extended-range", nargs="+", type=float, default=[0, -39.68, -3, 100, 39.68, 1])
    p.add_argument("--alphas", default="0,1e-6,1e-4,1e-3,1e-2,1e-1,1")
    p.add_argument("--max-train-frames", type=int, default=0)
    p.add_argument("--max-val-frames", type=int, default=0)
    p.add_argument("--model-kind", choices=["ridge", "mlp"], default="ridge")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--hidden-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


VEHICLE_TYPES = {"car", "van", "truck", "bus"}


def id_to_str(index: int) -> str:
    return f"{index:06d}"


def frame_id(path: str) -> str:
    return Path(path).stem


def get_3d_8points(size: list[float], yaw: float, center: list[float]) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    l, w, h = size
    corners = np.array(
        [
            [l / 2, w / 2, 0],
            [l / 2, -w / 2, 0],
            [-l / 2, -w / 2, 0],
            [-l / 2, w / 2, 0],
            [l / 2, w / 2, h],
            [l / 2, -w / 2, h],
            [-l / 2, -w / 2, h],
            [-l / 2, w / 2, h],
        ],
        dtype=np.float64,
    )
    return corners @ rot.T + np.asarray(center, dtype=np.float64)


def label_boxes(path: Path, prefer_world: bool) -> np.ndarray:
    labels = json.loads(path.read_text())
    boxes = []
    for label in labels:
        if str(label.get("type", "")).lower() not in VEHICLE_TYPES:
            continue
        if prefer_world and "world_8_points" in label:
            boxes.append(np.asarray(label["world_8_points"], dtype=np.float64))
            continue
        dims = label.get("3d_dimensions") or {}
        loc = label.get("3d_location") or {}
        try:
            l = float(dims["l"])
            w = float(dims["w"])
            h = float(dims["h"])
            x = float(loc["x"])
            y = float(loc["y"])
            z = float(loc["z"]) - h / 2.0
            rot = float(label.get("rotation", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if l == 0 or w == 0 or h == 0:
            continue
        boxes.append(get_3d_8points([l, w, h], rot, [x, y, z]))
    if not boxes:
        return np.zeros((0, 8, 3), dtype=np.float64)
    return np.stack(boxes).astype(np.float64)


def box_info(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    center = np.mean(boxes, axis=1)
    size = np.zeros((boxes.shape[0], 3), dtype=np.float64)
    size[:, 2] = np.max(boxes[:, :, 2], axis=1) - np.min(boxes[:, :, 2], axis=1)
    for i, box in enumerate(boxes):
        xy = box[:, :2] - center[i, :2]
        cov = xy.T @ xy
        vals, vecs = np.linalg.eigh(cov)
        main_axis = vecs[:, int(np.argmax(vals))]
        side_axis = np.array([-main_axis[1], main_axis[0]], dtype=np.float64)
        ext_main = xy @ main_axis
        ext_side = xy @ side_axis
        length = float(ext_main.max() - ext_main.min())
        width = float(ext_side.max() - ext_side.min())
        if width > length:
            length, width = width, length
        size[i, 0] = length
        size[i, 1] = width
    return center, size


def box_yaw(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float64)
    center = np.mean(boxes, axis=1)
    yaw = np.zeros((boxes.shape[0],), dtype=np.float64)
    for i, box in enumerate(boxes):
        xy = box[:, :2] - center[i, :2]
        cov = xy.T @ xy
        vals, vecs = np.linalg.eigh(cov)
        main_axis = vecs[:, int(np.argmax(vals))]
        yaw[i] = math.atan2(float(main_axis[1]), float(main_axis[0]))
    return yaw


def box_features(boxes: np.ndarray, horizon_frames: int) -> np.ndarray:
    n = len(boxes)
    if n == 0:
        return np.zeros((0, 18), dtype=np.float64)
    center, size = box_info(boxes)
    center = center.astype(np.float64)
    size = np.clip(size.astype(np.float64), 1e-3, None)
    yaw = box_yaw(boxes)
    h = np.full((n, 1), float(horizon_frames), dtype=np.float64)
    base = np.concatenate(
        [
            center[:, :2],
            center[:, 2:3],
            size,
            np.sin(2.0 * yaw)[:, None],
            np.cos(2.0 * yaw)[:, None],
            h,
        ],
        axis=1,
    )
    # Light nonlinear terms keep inference dependency-free while giving the
    # regressor enough capacity for horizon- and location-dependent motion.
    xy_h = base[:, [0, 1]] * h
    h2 = h * h
    inv_size = 1.0 / size[:, :2]
    return np.concatenate([base, xy_h, h2, base[:, :2] ** 2, base[:, 3:5] ** 2, inv_size], axis=1)


def fit_normalizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xb = np.concatenate([np.ones((x.shape[0], 1), dtype=x.dtype), x], axis=1)
    eye = np.eye(xb.shape[1], dtype=x.dtype)
    eye[0, 0] = 0.0
    return np.linalg.solve(xb.T @ xb + alpha * eye, xb.T @ y)


def predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    xb = np.concatenate([np.ones((x.shape[0], 1), dtype=x.dtype), x], axis=1)
    return xb @ weights


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    layers: list[nn.Module] = []
    in_dim = x_train.shape[1]
    for layer_idx in range(args.hidden_layers):
        layers.append(nn.Linear(in_dim if layer_idx == 0 else args.hidden_dim, args.hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(args.hidden_dim if args.hidden_layers else in_dim, 2))
    model = nn.Sequential(*layers).to(device)

    train_ds = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    xv = torch.from_numpy(x_val.astype(np.float32)).to(device)
    yv = torch.from_numpy(y_val.astype(np.float32)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    best_state = None
    best_rmse = float("inf")
    best_mae = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * xb.shape[0]
            train_seen += xb.shape[0]

        model.eval()
        with torch.no_grad():
            pred = model(xv)
            err = pred - yv
            rmse = float(torch.sqrt(torch.mean(err * err)).item())
            mae = float(torch.mean(torch.abs(err)).item())
        record = {"epoch": epoch, "train_loss": train_loss / max(train_seen, 1), "val_rmse": rmse, "val_mae": mae}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if rmse < best_rmse - 1e-6:
            best_rmse = rmse
            best_mae = mae
            best_state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("MLP training did not produce a checkpoint")

    arrays: dict[str, np.ndarray] = {}
    linear_idx = 0
    for module_idx, module in enumerate(model):
        if not isinstance(module, nn.Linear):
            continue
        arrays[f"layer{linear_idx}_weight"] = best_state[f"{module_idx}.weight"]
        arrays[f"layer{linear_idx}_bias"] = best_state[f"{module_idx}.bias"]
        linear_idx += 1
    metadata = {
        "model_kind": "mlp",
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "epochs_requested": args.epochs,
        "epochs_ran": len(history),
        "best_epoch": min(history, key=lambda r: r["val_rmse"])["epoch"],
        "val_rmse": best_rmse,
        "val_mae": best_mae,
        "history": history,
        "device": str(device),
    }
    return arrays, metadata


def transform_boxes(boxes: np.ndarray, root: Path, from_coord: str, to_coord: str, infra_name: str, veh_name: str) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    trans = Coord_transformation(from_coord, to_coord, str(root), infra_name, veh_name)
    return trans(boxes).astype(np.float64)


def load_infos(root: Path) -> tuple[dict[str, dict], dict[str, dict], list[dict]]:
    inf_infos = json.loads((root / "infrastructure-side/data_info.json").read_text())
    veh_infos = json.loads((root / "vehicle-side/data_info.json").read_text())
    coop_infos = json.loads((root / "cooperative/data_info.json").read_text())
    inf_map = {"infrastructure-side/" + info["pointcloud_path"]: info for info in inf_infos}
    veh_map = {"vehicle-side/" + info["pointcloud_path"]: info for info in veh_infos}
    return inf_map, veh_map, coop_infos


def split_coop_infos(coop_infos: list[dict], split_path: Path, split: str) -> list[dict]:
    split_ids = set(json.loads(split_path.read_text())["cooperative_split"][split])
    return [info for info in coop_infos if frame_id(info["vehicle_image_path"]) in split_ids]


def matched_examples(
    root: Path,
    inf_map: dict[str, dict],
    veh_map: dict[str, dict],
    coop_infos: list[dict],
    horizon: int,
    max_match_dist: float,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    frames_seen = 0
    for info in coop_infos:
        if max_frames and frames_seen >= max_frames:
            break
        frames_seen += 1
        cur_inf = inf_map.get(info["infrastructure_pointcloud_path"])
        cur_veh = veh_map.get(info["vehicle_pointcloud_path"])
        if cur_inf is None or cur_veh is None:
            continue
        cur_inf_id = int(frame_id(cur_inf["pointcloud_path"]))
        if cur_inf_id - horizon < int(cur_inf["batch_start_id"]):
            continue
        old_key = "infrastructure-side/velodyne/" + id_to_str(cur_inf_id - horizon) + ".pcd"
        old_inf = inf_map.get(old_key)
        if old_inf is None:
            continue

        old_path = root / "infrastructure-side" / old_inf["label_lidar_std_path"]
        cur_path = root / info["cooperative_label_path"]
        old_boxes = label_boxes(old_path, prefer_world=False)
        cur_boxes = label_boxes(cur_path, prefer_world=True)
        old_boxes = transform_boxes(
            old_boxes,
            root,
            "Infrastructure_lidar",
            "Vehicle_lidar",
            frame_id(old_inf["image_path"]),
            frame_id(cur_veh["image_path"]),
        )
        cur_boxes = transform_boxes(
            cur_boxes,
            root,
            "World",
            "Vehicle_lidar",
            frame_id(cur_inf["image_path"]),
            frame_id(cur_veh["image_path"]),
        )
        if len(old_boxes) == 0 or len(cur_boxes) == 0:
            continue
        old_center, _ = box_info(old_boxes)
        cur_center, _ = box_info(cur_boxes)
        dist = np.linalg.norm(old_center[:, None, :2] - cur_center[None, :, :2], axis=2)
        rows, cols = linear_sum_assignment(dist)
        keep = dist[rows, cols] <= max_match_dist
        if not np.any(keep):
            continue
        rows = rows[keep]
        cols = cols[keep]
        xs.append(box_features(old_boxes, horizon)[rows])
        ys.append(cur_center[cols, :2] - old_center[rows, :2])
    if not xs:
        return np.zeros((0, 18), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def collect_split(args: argparse.Namespace, split: str, horizons: list[int], max_frames: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    inf_map, veh_map, coop_infos_all = load_infos(args.dair_root)
    coop_infos = split_coop_infos(coop_infos_all, args.split_data_path, split)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for horizon in horizons:
        x_h, y_h = matched_examples(args.dair_root, inf_map, veh_map, coop_infos, horizon, args.max_match_dist, max_frames)
        counts[str(horizon)] = int(len(y_h))
        if len(y_h):
            xs.append(x_h)
            ys.append(y_h)
    if not xs:
        raise RuntimeError(f"no matched examples for split={split}")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), counts


def main() -> None:
    args = parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    x_train, y_train, train_counts = collect_split(args, "train", horizons, args.max_train_frames)
    x_val, y_val, val_counts = collect_split(args, "val", horizons, args.max_val_frames)

    mean, std = fit_normalizer(x_train)
    xt = (x_train - mean) / std
    xv = (x_val - mean) / std

    if args.model_kind == "ridge":
        best = None
        evals = []
        for alpha in alphas:
            weights = ridge_fit(xt, y_train, alpha)
            pred = predict(xv, weights)
            rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
            mae = float(np.mean(np.abs(pred - y_val)))
            evals.append({"alpha": alpha, "val_rmse": rmse, "val_mae": mae})
            if best is None or rmse < best[0]:
                best = (rmse, mae, alpha, weights)
        assert best is not None
        rmse, mae, alpha, weights = best
        model_arrays = {"weights": weights}
        model_metadata = {
            "model_kind": "ridge",
            "selected_alpha": alpha,
            "val_rmse": rmse,
            "val_mae": mae,
            "evals": evals,
        }
    else:
        model_arrays, model_metadata = train_mlp(xt, y_train, xv, y_val, args)
        rmse = float(model_metadata["val_rmse"])
        mae = float(model_metadata["val_mae"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": "dair_ours_motion_compensator",
        "feature_dim": int(mean.shape[0]),
        "horizons": horizons,
        "train_examples": int(len(y_train)),
        "val_examples": int(len(y_val)),
        "train_counts_by_horizon": train_counts,
        "val_counts_by_horizon": val_counts,
        "max_match_dist": args.max_match_dist,
        **model_metadata,
    }
    np.savez(args.out, mean=mean, std=std, metadata=json.dumps(metadata, sort_keys=True), **model_arrays)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
