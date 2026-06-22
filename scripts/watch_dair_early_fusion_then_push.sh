#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/xuyifan/v2x_code_ckpt}"
CLEAN_UPLOAD="${CLEAN_UPLOAD:-/tmp/coperception_clean_upload}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/dair_v2x/official_baselines}"
POLL_SECONDS="${POLL_SECONDS:-600}"
BRANCH="${BRANCH:-master}"

cd "${ROOT}"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*"
}

all_complete_logs_present() {
  local k log_file
  for k in 3 4 5; do
    log_file="${RESULT_ROOT}/logs/early_fusion_k${k}.log"
    if ! grep -q "Average Communication Cost" "${log_file}" 2>/dev/null; then
      return 1
    fi
  done
  return 0
}

static_latex_check() {
  python - <<'PY'
import re
from pathlib import Path

root = Path("DLPCM")
tex_files = [root / "main.tex", root / "4_experiment.tex"]
missing = []
for tex in tex_files:
    text = tex.read_text()
    for match in re.finditer(r"\\input\{([^}]+)\}", text):
        raw = root / match.group(1)
        candidates = [raw] if raw.suffix else [Path(str(raw) + ".tex")]
        if not any(path.exists() for path in candidates):
            missing.append((str(tex), "input", str(raw)))
    for match in re.finditer(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
        raw = root / match.group(1)
        candidates = [Path(str(raw) + ext) for ext in ("", ".pdf", ".png", ".jpg", ".jpeg")]
        if not any(path.exists() for path in candidates):
            missing.append((str(tex), "graphic", str(raw)))
for tex in root.glob("*.tex"):
    text = tex.read_text(errors="ignore")
    begins = re.findall(r"\\begin\{([^}]+)\}", text)
    ends = re.findall(r"\\end\{([^}]+)\}", text)
    if sorted(begins) != sorted(ends):
        missing.append((str(tex), "begin_end_mismatch", f"begins={begins} ends={ends}"))
if missing:
    for item in missing:
        print(item)
    raise SystemExit(1)
print("static LaTeX path check passed")
PY
}

sync_curated_content() {
  local src="${ROOT}"
  local dst="${CLEAN_UPLOAD}"

  [[ -d "${dst}/.git" ]] || {
    log "clean upload repo missing: ${dst}"
    return 2
  }

  cd "${dst}"
  git pull --ff-only origin "${BRANCH}"
  cd "${src}"

  for file in .gitattributes .gitignore README.md requirements.txt; do
    rsync -a "${file}" "${dst}/${file}"
  done
  rsync -a --delete --exclude='__pycache__/' --exclude='*.pyc' scripts/ "${dst}/scripts/"
  rsync -a --delete docs/ "${dst}/docs/"
  rsync -a --delete DLPCM/ "${dst}/DLPCM/"

  mkdir -p \
    "${dst}/results/dair_v2x/official_baselines" \
    "${dst}/results/dair_v2x/summary" \
    "${dst}/results/dair_v2x/tables" \
    "${dst}/results/dair_v2x/our_method" \
    "${dst}/results/revision_tables" \
    "${dst}/results/revision_figures" \
    "${dst}/results/summary" \
    "${dst}/results/summary_revision"

  rsync -a results/dair_v2x/official_baselines/summary_partial.csv \
    "${dst}/results/dair_v2x/official_baselines/summary_partial.csv"
  rsync -a --delete results/dair_v2x/summary/ "${dst}/results/dair_v2x/summary/"
  rsync -a --delete results/dair_v2x/tables/ "${dst}/results/dair_v2x/tables/"
  for dir in current_T10_n5_ckpt11000 final_T10_n5_ckpt16000 full_T10_n10_ckpt15000; do
    rsync -a --delete "results/dair_v2x/our_method/${dir}/" \
      "${dst}/results/dair_v2x/our_method/${dir}/"
  done
  rsync -a results/dair_v2x/our_method/delay_grid_T10_n5_ckpt10000_latency_csv_smoke.csv \
    "${dst}/results/dair_v2x/our_method/"
  rsync -a results/dair_v2x/robustness_T10_n10_ckpt15000.csv \
    results/dair_v2x/robustness_T10_n5_ckpt16000.csv \
    "${dst}/results/dair_v2x/"
  rsync -a --delete results/revision_tables/ "${dst}/results/revision_tables/"
  rsync -a --delete results/revision_figures/ "${dst}/results/revision_figures/"
  rsync -a --delete results/summary/ "${dst}/results/summary/"
  rsync -a --delete results/summary_revision/ "${dst}/results/summary_revision/"
}

log "watching early_fusion_k3/k4/k5 for final refresh and push"
until all_complete_logs_present; do
  log "not complete yet; sleeping ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

log "all early-fusion logs complete; refreshing artifacts"
python scripts/parse_dair_official_baseline_logs.py \
  --log-dir "${RESULT_ROOT}/logs" \
  --out "${RESULT_ROOT}/summary_partial.csv"
python scripts/make_dair_revision_tables.py
python scripts/make_dair_paper_artifacts.py

python - <<'PY'
from pathlib import Path

path = Path("results/dair_v2x/summary/current_status.csv")
text = path.read_text()
text = text.replace("Optional DAIR early-fusion extension,running", "Optional DAIR early-fusion extension,completed")
text = text.replace(
    "Active experiment processes,running,nvidia-smi,wait for optional early-fusion extension to finish",
    "Active experiment processes,none,nvidia-smi,none",
)
text = text.replace(
    "completed runs include veh_only_k0, inf_only_k0, late_fusion_tclf_k0..k5, late_fusion_no_comp_k1..k5, early_fusion_k0..k2",
    "completed runs include veh_only_k0, inf_only_k0, late_fusion_tclf_k0..k5, late_fusion_no_comp_k1..k5, early_fusion_k0..k5",
)
path.write_text(text)
PY

static_latex_check
sync_curated_content

cd "${CLEAN_UPLOAD}"
if [[ -n "$(git status --short)" ]]; then
  git add -A
  git commit -m "Complete DAIR early fusion delay extension"
  git push origin "${BRANCH}"
  log "pushed final early-fusion update: $(git rev-parse HEAD)"
else
  log "no curated changes to push"
fi
