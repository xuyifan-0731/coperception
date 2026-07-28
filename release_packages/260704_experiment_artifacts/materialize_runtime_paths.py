#!/usr/bin/env python3
"""Materialize restored 260704 artifacts into runtime paths.

Run after:
    python restore_split_checkpoints.py

Example:
    python materialize_runtime_paths.py --repo-root /raid/xuyifan/v2x_code_ckpt \
      --traf-root /tmp/TraF-Align_partial --mode symlink
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing package artifact: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def add_file(mappings: list[tuple[Path, Path]], package_rel: str, runtime_path: Path) -> None:
    mappings.append((PACKAGE_ROOT / package_rel, runtime_path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--traf-root", type=Path, default=Path("/tmp/TraF-Align_partial"))
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    traf = args.traf_root.resolve()
    mappings: list[tuple[Path, Path]] = []

    # Baseline checkpoints/configs used by the runbook.
    add_file(
        mappings,
        "01_v2xsim/DATA/checkpoints/baseline_DATA/net_epoch_bestval_at19.pth",
        repo / "external/DATA/opencood/logs/DATA_v2xsim_pipeline_2026_06_10_14_27_14/net_epoch_bestval_at19.pth",
    )
    add_file(
        mappings,
        "02_dair_v2x_c/DATA/checkpoints/baseline_DATA/net_epoch_bestval_at21.pth",
        repo / "runs/data_dair_official_eval/net_epoch_bestval_at21.pth",
    )
    add_file(
        mappings,
        "02_dair_v2x_c/DATA/checkpoints/baseline_DATA/config.yaml",
        repo / "runs/data_dair_official_eval/config.yaml",
    )
    add_file(
        mappings,
        "02_dair_v2x_c/LRCP/checkpoints/baseline_LRCP_stage2/net_epoch_bestval_at1.pth",
        repo / "external/LRCP/opencood/logs/dairv2x_lrcp_point_pillar_lidar_deformable_attn_stage2_2026_07_01_19_32_46/net_epoch_bestval_at1.pth",
    )
    add_file(
        mappings,
        "03_dair_v2x_seq/TraF-Align/checkpoints/baseline_official_trafalign/net_epoch60.pth",
        traf / "checkpoints/dair_v2x_seq_Trafalign_2024_12_28_04_42_36/net_epoch60.pth",
    )
    add_file(
        mappings,
        "03_dair_v2x_seq/TraF-Align/checkpoints/baseline_official_trafalign/config.yaml",
        traf / "checkpoints/dair_v2x_seq_Trafalign_2024_12_28_04_42_36/config.yaml",
    )

    # Ours checkpoints used by current 260704 tables and diagnostics.
    ours_files = [
        ("01_v2xsim/DATA/checkpoints/ours/v2xsim_data_ifam_joint_delay_comp.pth", "v2xsim_data_ifam_joint_delay_comp.pth"),
        ("01_v2xsim/DATA/checkpoints/ours/v2xsim_data_ifam_joint_delay_comp_scale01.pth", "v2xsim_data_ifam_joint_delay_comp_scale01.pth"),
        ("01_v2xsim/V2VNet/checkpoints/ours/v2xsim_ours_prediction_model.pth", "v2xsim_ours_prediction_model.pth"),
        ("02_dair_v2x_c/official_late_fusion/checkpoints/ours_rf2/dair_v2x_late_fusion_rf2_ours.pkl", "dair_v2x_late_fusion_rf2_ours.pkl"),
        ("02_dair_v2x_c/official_late_fusion/checkpoints/ours_rf2/dair_v2x_late_fusion_rf2_ours_trainonly.pkl", "dair_v2x_late_fusion_rf2_ours_trainonly.pkl"),
        ("02_dair_v2x_c/official_late_fusion/checkpoints/ours_rf2/dair_v2x_late_fusion_rf2_ours_trainval_diagnostic.pkl", "dair_v2x_late_fusion_rf2_ours_trainval_diagnostic.pkl"),
        ("02_dair_v2x_c/DATA/checkpoints/ours/dair_data_ifam_feature_t10n10_ours.npz", "dair_data_ifam_feature_t10n10_ours.npz"),
        ("02_dair_v2x_c/DATA/checkpoints/ours/dair_data_ifam_joint_delay_comp.pth", "dair_data_ifam_joint_delay_comp.pth"),
        ("02_dair_v2x_c/DATA/checkpoints/ours/dair_data_ifam_joint_delay_comp_unfreeze_ifam_head_lr1e5_1500.pth", "dair_data_ifam_joint_delay_comp_unfreeze_ifam_head_lr1e5_1500.pth"),
        ("02_dair_v2x_c/DATA/checkpoints/ours/dair_data_ifam_joint_delay_comp_unfreeze_ifam_head_lowcomp_weighted_lr1e5_1500.pth", "dair_data_ifam_joint_delay_comp_unfreeze_ifam_head_lowcomp_weighted_lr1e5_1500.pth"),
        ("03_dair_v2x_seq/TraF-Align/checkpoints/ours_feature_t10n10/dair_v2x_seq_trafalign_feature_t10n10_ours_spconv2121.npz", "dair_v2x_seq_trafalign_feature_t10n10_ours_spconv2121.npz"),
    ]
    for package_rel, runtime_name in ours_files:
        add_file(mappings, package_rel, repo / "checkpoints/curated/ours_models" / runtime_name)

    traf_dirs = [
        "trafalign_joint_delay_comp_only_spconv2121",
        "trafalign_joint_delay_comp_only_adaptive2to4_spconv2121",
        "trafalign_joint_delay_comp_only_adaptive2to4_bypass_comp0_spconv2121",
    ]
    for name in traf_dirs:
        for file_name in ["net_epoch1.pth", "config.yaml", "train_metrics.csv"]:
            add_file(
                mappings,
                f"03_dair_v2x_seq/TraF-Align/checkpoints/ours_joint_delay_comp/{name}/{file_name}",
                repo / "checkpoints/curated/ours_models" / name / file_name,
            )

    diag_dirs = [
        "trafalign_joint_delay_comp_spconv2121",
        "trafalign_joint_unfreeze_tail_head_20260722",
        "trafalign_joint_unfreeze_field_offset_20260722",
        "trafalign_joint_unfreeze_field_offset_gate06_20260722",
        "trafalign_joint_unfreeze_tail_head_gate06_20260722",
    ]
    for name in diag_dirs:
        for file_name in ["net_epoch1.pth", "config.yaml", "train_metrics.csv"]:
            src = PACKAGE_ROOT / f"03_dair_v2x_seq/TraF-Align/checkpoints/diagnostic_unfreeze/{name}/{file_name}"
            if src.exists():
                mappings.append((src, repo / "checkpoints/curated/ours_models" / name / file_name))

    completed = 0
    for src, dst in mappings:
        link_or_copy(src, dst, args.mode, args.overwrite)
        completed += 1
    print(f"Materialized {completed} artifact paths using mode={args.mode}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
