#!/usr/bin/env python3
"""End-to-end DATA reproduction pipeline on V2X-Sim 2.0.

The script is intentionally resumable.  Re-running the same command skips
finished stages and continues from the first missing artifact.
"""

import argparse
import csv
import glob
import json
import os
import shlex
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "external" / "DATA"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "data_v2xsim_pipeline"
ZIP_NAMES = ["v2.0.zip", "lidar1.zip", "lidar2.zip", "lidar3.zip", "maps.zip", "lidarseg.zip"]
HF_REPO = "ai4ce-drive/V2X-Sim-2.0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all", "download", "extract", "info", "config",
                                 "smoke", "train", "eval", "summarize"])
    parser.add_argument("--conda-env", default="dair-v2x-openmmlab")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--zip-dir", default=str(REPO_ROOT / "datasets" / "V2X-Sim-2.0_zips"))
    parser.add_argument("--raw-dir", default=str(REPO_ROOT / "datasets" / "V2X-Sim-2.0"))
    parser.add_argument("--info-dir", default=str(REPO_ROOT / "datasets" / "v2xsim2_info"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Reserved for future train.py patching; DATA train.py currently uses 4.")
    parser.add_argument("--delay-grid", default="0:0,0:2,0:4,1:0,1:2,1:4,2:0,2:2,2:4,3:0,3:2,3:4,4:0,4:2,4:4,5:0,5:2,5:4",
                        help="comm:comp pairs in frames. Default follows Table-2 style comm 0-5 and comp 0/2/4.")
    parser.add_argument("--download-missing", action="store_true",
                        help="Download missing V2X-Sim zips from HuggingFace if needed.")
    parser.add_argument("--verify-zips", action="store_true",
                        help="Run full zip integrity checks before extraction. Slow for 45GB data.")
    parser.add_argument("--force", action="store_true",
                        help="Run selected stage even when status says done.")
    return parser.parse_args()


def load_status(run_dir):
    path = Path(run_dir) / "status.json"
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {"stages": {}, "artifacts": {}}


def save_status(run_dir, status):
    path = Path(run_dir) / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(status, f, indent=2, sort_keys=True)
    tmp.replace(path)


def should_run(args, status, stage):
    return args.force or status["stages"].get(stage) != "done"


def mark(status, run_dir, stage, value="done", **artifacts):
    status["stages"][stage] = value
    status["artifacts"].update({k: str(v) for k, v in artifacts.items() if v is not None})
    save_status(run_dir, status)


def log_path(run_dir, name):
    path = Path(run_dir) / "logs" / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_cmd(cmd, run_dir, name, env=None, cwd=None):
    log = log_path(run_dir, name)
    printable = " ".join(shlex.quote(str(x)) for x in cmd)
    print(f"[{name}] {printable}")
    print(f"[{name}] log: {log}")
    with log.open("w") as f:
        f.write(f"$ {printable}\n")
        f.flush()
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd or REPO_ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"{name} failed with exit code {ret}. See {log}")
    return log


def conda_cmd(args, command):
    return ["conda", "run", "--no-capture-output", "-n", args.conda_env] + list(command)


def command_env(args):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{DATA_ROOT}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return env


def download_missing(args, status):
    zip_dir = Path(args.zip_dir)
    zip_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in ZIP_NAMES if not (zip_dir / name).exists()]
    if not missing:
        mark(status, args.run_dir, "download", zip_dir=zip_dir)
        return
    if not args.download_missing:
        raise FileNotFoundError(
            "Missing V2X-Sim zip files: "
            + ", ".join(missing)
            + ". Re-run with --download-missing, or put them in "
            + str(zip_dir)
        )

    script = (
        "from huggingface_hub import hf_hub_download\n"
        f"repo={HF_REPO!r}\n"
        f"local_dir={str(zip_dir)!r}\n"
        f"files={missing!r}\n"
        "for name in files:\n"
        "    print('downloading', name)\n"
        "    hf_hub_download(repo_id=repo, repo_type='dataset', filename=name, "
        "local_dir=local_dir, local_dir_use_symlinks=False, resume_download=True)\n"
    )
    run_cmd(conda_cmd(args, ["python", "-c", script]), args.run_dir, "download_v2xsim_zips", command_env(args))
    mark(status, args.run_dir, "download", zip_dir=zip_dir)


def zip_is_valid(path):
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
        return bad is None
    except zipfile.BadZipFile:
        return False


def extract_zips(args, status):
    raw_dir = Path(args.raw_dir)
    zip_dir = Path(args.zip_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in ZIP_NAMES:
        zpath = zip_dir / name
        if not zpath.exists():
            raise FileNotFoundError(zpath)
        if args.verify_zips and not zip_is_valid(zpath):
            raise RuntimeError(f"Corrupt zip file: {zpath}")

    marker = raw_dir / ".extract_complete"
    if marker.exists() and not args.force:
        mark(status, args.run_dir, "extract", raw_dir=raw_dir)
        return

    for name in ZIP_NAMES:
        run_cmd(["unzip", "-qn", str(zip_dir / name), "-d", str(raw_dir)],
                args.run_dir, f"extract_{name}", command_env(args))
    if not (raw_dir / "v2.0" / "scene.json").exists():
        raise RuntimeError(f"Extraction finished but {raw_dir}/v2.0/scene.json is missing")
    mini_meta = raw_dir / "v1.0-mini"
    if not mini_meta.exists():
        mini_meta.symlink_to(raw_dir / "v2.0", target_is_directory=True)
    det_meta = raw_dir / "v2xsim-det"
    det_meta.mkdir(exist_ok=True)
    for src in (raw_dir / "v2.0").glob("*.json"):
        if src.name == "lidarseg.json":
            continue
        dst = det_meta / src.name
        if not dst.exists():
            dst.symlink_to(src)
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    mark(status, args.run_dir, "extract", raw_dir=raw_dir)


def generate_info(args, status):
    info_dir = Path(args.info_dir)
    expected = [info_dir / f"v2xsim_infos_{split}.pkl" for split in ("train", "val", "test")]
    if all(path.exists() for path in expected) and not args.force:
        mark(status, args.run_dir, "info", info_dir=info_dir)
        return
    cmd = conda_cmd(args, [
        "python", str(REPO_ROOT / "scripts" / "prepare_v2xsim2_info_for_data.py"),
        "--dataroot", str(Path(args.raw_dir)),
        "--output-dir", str(info_dir),
        "--skip-existing",
    ])
    run_cmd(cmd, args.run_dir, "generate_v2xsim_info", command_env(args))
    mark(status, args.run_dir, "info", info_dir=info_dir)


def data_v2xsim_config(args):
    info = Path(args.info_dir)
    cav = [-32, -32, -3, 32, 32, 1]
    return {
        "name": "DATA_v2xsim_pipeline",
        "root_dir": str(info / "v2xsim_infos_train.pkl"),
        "validate_dir": str(info / "v2xsim_infos_val.pkl"),
        "test_dir": str(info / "v2xsim_infos_test.pkl"),
        "yaml_parser": "load_general_params",
        "train_params": {
            "batch_size": args.batch_size,
            "epoches": args.epochs,
            "eval_freq": 1,
            "save_freq": 1,
            "max_cav": 5,
            "single_weight": 1,
        },
        "comm_range": 70,
        "input_source": ["lidar"],
        "label_type": "lidar",
        "cav_lidar_range": cav,
        "heter": {
            "assignment_path": None,
            "ego_modality": "m1",
            "mapping_dict": {"m1": "m1", "m2": "m1", "m3": "m1", "m4": "m1"},
            "modality_setting": {
                "m1": {
                    "sensor_type": "lidar",
                    "core_method": "point_pillar",
                    "preprocess": {
                        "core_method": "SpVoxelPreprocessor",
                        "args": {
                            "voxel_size": [0.4, 0.4, 4],
                            "max_points_per_voxel": 32,
                            "max_voxel_train": 32000,
                            "max_voxel_test": 70000,
                        },
                        "cav_lidar_range": cav,
                    },
                }
            },
        },
        "fusion": {
            "core_method": "intermediateheter",
            "dataset": "v2xsim",
            "args": {"proj_first": False, "grid_conf": None, "data_aug_conf": None},
        },
        "data_augment": [
            {"NAME": "random_world_flip", "ALONG_AXIS_LIST": ["x"]},
            {"NAME": "random_world_rotation", "WORLD_ROT_ANGLE": [-0.78539816, 0.78539816]},
            {"NAME": "random_world_scaling", "WORLD_SCALE_RANGE": [0.95, 1.05]},
        ],
        "preprocess": {
            "core_method": "SpVoxelPreprocessor",
            "args": {
                "voxel_size": [0.4, 0.4, 4],
                "max_points_per_voxel": 1,
                "max_voxel_train": 1,
                "max_voxel_test": 1,
            },
            "cav_lidar_range": cav,
        },
        "postprocess": {
            "core_method": "VoxelPostprocessor",
            "gt_range": cav,
            "anchor_args": {
                "cav_lidar_range": cav,
                "l": 3.9,
                "w": 1.6,
                "h": 1.56,
                "r": [0, 90],
                "feature_stride": 2,
                "num": 2,
            },
            "target_args": {"pos_threshold": 0.6, "neg_threshold": 0.45, "score_threshold": 0.2},
            "order": "hwl",
            "max_num": 150,
            "nms_thresh": 0.15,
            "dir_args": {"dir_offset": 0.7853, "num_bins": 2, "anchor_yaw": [0, 90]},
        },
        "model": {
            "core_method": "DATA",
            "args": {
                "lidar_range": cav,
                "supervise_single": True,
                "m1": {
                    "core_method": "point_pillar",
                    "sensor_type": "lidar",
                    "encoder_args": {
                        "voxel_size": [0.4, 0.4, 4],
                        "lidar_range": cav,
                        "pillar_vfe": {
                            "use_norm": True,
                            "with_distance": False,
                            "use_absolute_xyz": True,
                            "num_filters": [64],
                        },
                        "point_pillar_scatter": {"num_features": 64},
                    },
                    "backbone_args": {
                        "layer_nums": [3, 5, 5],
                        "layer_strides": [2, 2, 2],
                        "num_filters": [64, 128, 256],
                        "upsample_strides": [1, 2, 4],
                        "num_upsample_filter": [128, 128, 128],
                    },
                    "aligner_args": {"core_method": "identity"},
                },
                "fusion_backbone": {
                    "resnext": False,
                    "layer_nums": [1],
                    "layer_strides": [1],
                    "num_filters": [384],
                    "upsample_strides": [1],
                    "num_upsample_filter": [1],
                    "anchor_number": 2,
                    "in_channels": 384,
                    "spatial_map_h": 80,
                    "spatial_map_w": 80,
                    "spatial_map_x_range": [-32, 32],
                    "spatial_map_y_range": [-32, 32],
                },
                "in_head": 384,
                "anchor_number": 2,
                "dir_args": {"dir_offset": 0.7853, "num_bins": 2, "anchor_yaw": [0, 90]},
            },
        },
        "loss": {
            "core_method": "point_pillar_pyramid_loss",
            "args": {
                "pos_cls_weight": 2.0,
                "cls": {"type": "SigmoidFocalLoss", "alpha": 0.25, "gamma": 2.0, "weight": 1.0},
                "reg": {"type": "WeightedSmoothL1Loss", "sigma": 3.0, "codewise": True, "weight": 2.0},
                "dir": {"type": "WeightedSoftmaxClassificationLoss", "weight": 0.2,
                        "args": {"dir_offset": 0.7853, "num_bins": 2, "anchor_yaw": [0, 90]}},
                "depth": {"weight": 1.0},
                "pyramid": {"relative_downsample": [1], "weight": [0.4]},
                "domain": {"weight": 1.0, "batchsize": args.batch_size},
            },
        },
        "optimizer": {"core_method": "Adam", "lr": 0.002,
                      "args": {"eps": 1e-10, "weight_decay": 1e-4}},
        "lr_scheduler": {"core_method": "multistep", "gamma": 0.1, "step_size": [15, 30]},
    }


def write_config(args, status):
    config_dir = Path(args.run_dir) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "data_v2xsim.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(data_v2xsim_config(args), f, sort_keys=False)
    mark(status, args.run_dir, "config", config_path=config_path)
    print(f"[config] wrote {config_path}")


def smoke_test(args, status):
    if not should_run(args, status, "smoke"):
        return
    config_path = status["artifacts"]["config_path"]
    script = f"""
import torch
from torch.utils.data import DataLoader
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
hypes = yaml_utils.load_yaml({config_path!r})
dataset = build_dataset(hypes, visualize=False, train=False)
batch = dataset.collate_batch_test([dataset[0]])
assert batch is not None
model = train_utils.create_model(hypes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
batch = train_utils.to_device(batch, device)
with torch.no_grad():
    out = model(batch['ego'])
print('smoke ok', out['cls_preds'].shape, out['reg_preds'].shape)
"""
    run_cmd(conda_cmd(args, ["python", "-c", script]), args.run_dir, "smoke_test", command_env(args), DATA_ROOT)
    mark(status, args.run_dir, "smoke")


def find_latest_model_dir():
    logs = DATA_ROOT / "opencood" / "logs"
    candidates = sorted(
        [Path(p) for p in glob.glob(str(logs / "DATA_v2xsim_pipeline_*"))],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def checkpoint_done(model_dir, epochs):
    if not model_dir:
        return False
    return (Path(model_dir) / f"net_epoch{epochs}.pth").exists()


def train(args, status):
    existing = status["artifacts"].get("model_dir")
    if checkpoint_done(existing, args.epochs) and not args.force:
        mark(status, args.run_dir, "train", model_dir=existing)
        return
    config_path = status["artifacts"]["config_path"]
    cmd = conda_cmd(args, ["python", "opencood/tools/train.py", "-y", config_path,
                           "--fusion_method", "intermediate"])
    if existing and Path(existing).exists():
        cmd.extend(["--model_dir", existing])
    run_cmd(cmd, args.run_dir, "train_data_v2xsim", command_env(args), DATA_ROOT)
    model_dir = find_latest_model_dir()
    if model_dir is None:
        raise RuntimeError("Training finished but no DATA_v2xsim_pipeline_* log dir was found")
    mark(status, args.run_dir, "train", model_dir=model_dir)


def parse_delay_grid(text):
    pairs = []
    for item in text.split(","):
        comm, comp = item.split(":")
        pairs.append((int(comm), int(comp)))
    return pairs


def eval_one(args, status, comm, comp):
    model_dir = Path(status["artifacts"]["model_dir"])
    name = f"eval_comm{comm}_comp{comp}"
    if status["stages"].get(name) == "done" and not args.force:
        return
    cmd = conda_cmd(args, [
        "python", "opencood/tools/inference.py",
        "--model_dir", str(model_dir),
        "--fusion_method", "intermediate",
        "--range", "32,32",
        "--save_vis_interval", "1000000",
    ])
    if comm or comp:
        cmd.extend(["--comm_delay", str(comm), "--comp_delay", str(comp)])
    before = set(model_dir.glob("eval_intermediate*.yaml"))
    run_cmd(cmd, args.run_dir, name, command_env(args), DATA_ROOT)
    after = set(model_dir.glob("eval_intermediate*.yaml"))
    new_files = sorted(after - before, key=lambda p: p.stat().st_mtime)
    eval_file = new_files[-1] if new_files else sorted(after, key=lambda p: p.stat().st_mtime)[-1]
    mark(status, args.run_dir, name, **{name: eval_file})


def evaluate(args, status):
    if "model_dir" not in status["artifacts"]:
        raise RuntimeError("No model_dir in status. Run train stage first.")
    for comm, comp in parse_delay_grid(args.delay_grid):
        eval_one(args, status, comm, comp)
    mark(status, args.run_dir, "eval")


def read_eval_yaml(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    return {
        "AP@0.3": float(data.get("ap30", 0)),
        "AP@0.5": float(data.get("ap_50", 0)),
        "AP@0.7": float(data.get("ap_70", 0)),
    }


def summarize(args, status):
    result_dir = Path(args.run_dir) / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for comm, comp in parse_delay_grid(args.delay_grid):
        key = f"eval_comm{comm}_comp{comp}"
        path = status["artifacts"].get(key)
        if not path or not Path(path).exists():
            continue
        metrics = read_eval_yaml(path)
        rows.append({
            "method": "DATA",
            "dataset": "V2X-Sim 2.0",
            "comm_delay_frames": comm,
            "comp_delay_frames": comp,
            "total_remote_delay_frames": comm + comp,
            **metrics,
            "eval_file": path,
        })

    csv_path = result_dir / "data_v2xsim_latency_ap.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "method", "dataset", "comm_delay_frames", "comp_delay_frames",
            "total_remote_delay_frames", "AP@0.3", "AP@0.5", "AP@0.7", "eval_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = result_dir / "data_v2xsim_latency_ap.md"
    with md_path.open("w") as f:
        f.write("# DATA on V2X-Sim 2.0 latency AP\n\n")
        f.write("大白话说明：这个表是在 V2X-Sim 2.0 上训练 DATA 后评测的 3D detection AP。"
                "通信时延是别的车传过来的信息晚了几帧，计算时延是本车/远端处理又晚了几帧；"
                "评测时 GT 仍用当前帧，输入用旧帧。\n\n")
        f.write("| 方法 | 数据集 | 通信时延/帧 | 计算时延/帧 | 远端总旧帧数 | AP@0.3 | AP@0.5 | AP@0.7 |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['method']} | {row['dataset']} | {row['comm_delay_frames']} | "
                f"{row['comp_delay_frames']} | {row['total_remote_delay_frames']} | "
                f"{row['AP@0.3']:.4f} | {row['AP@0.5']:.4f} | {row['AP@0.7']:.4f} |\n")

    progress_md = REPO_ROOT / "260604目前进度.md"
    block = (
        "\n\n## DATA 在 V2X-Sim 2.0 上的自动复现实验流水线\n\n"
        f"- 脚本：`scripts/run_data_v2xsim_pipeline.py`\n"
        f"- 配置：`{status['artifacts'].get('config_path', '')}`\n"
        f"- checkpoint 目录：`{status['artifacts'].get('model_dir', '')}`\n"
        f"- AP 汇总 CSV：`{csv_path}`\n"
        f"- AP 汇总 Markdown：`{md_path}`\n"
        "- 时延定义：通信时延 `comm_delay` 表示其他车的信息晚几帧；计算时延 "
        "`comp_delay` 表示处理过程再晚几帧。评测时 ego 输入取 `comp_delay` 旧帧，"
        "其他车输入取 `comm_delay + comp_delay` 旧帧，GT 仍然按当前帧算。\n"
    )
    with progress_md.open("a") as f:
        f.write(block)
    mark(status, args.run_dir, "summarize", csv_path=csv_path, md_path=md_path)


def main():
    args = parse_args()
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    status = load_status(args.run_dir)

    stage_order = ["download", "extract", "info", "config", "smoke", "train", "eval", "summarize"]
    selected = stage_order if args.stage == "all" else [args.stage]
    for stage in selected:
        if stage == "download" and should_run(args, status, stage):
            download_missing(args, status)
        elif stage == "extract" and should_run(args, status, stage):
            extract_zips(args, status)
        elif stage == "info" and should_run(args, status, stage):
            generate_info(args, status)
        elif stage == "config":
            write_config(args, status)
        elif stage == "smoke":
            smoke_test(args, status)
        elif stage == "train":
            train(args, status)
        elif stage == "eval":
            evaluate(args, status)
        elif stage == "summarize":
            summarize(args, status)
    print(f"Done. Status: {Path(args.run_dir) / 'status.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
