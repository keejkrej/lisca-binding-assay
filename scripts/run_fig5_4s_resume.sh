#!/usr/bin/env bash
# Resume fig5 4 s pipeline after interruption (skips completed ROI stacks).
set -euo pipefail

SRC_DATA=/home/jack/data/lisca_review/fig5/20260324_1
DATA=/home/jack/data/lisca_review/fig5/20260324_1_4s
BINDING=/home/jack/workspace/lisca-binding-assay
TIME_INTERVAL=4
LOG="$DATA/pipeline_4s.log"

cd "$BINDING"

for sub in align bbox roi; do
  if [[ ! -e "$DATA/$sub" ]]; then
    ln -sfn "$SRC_DATA/$sub" "$DATA/$sub"
  fi
done

{
  echo "==> $(date -Iseconds) Resume: rebuild ROI stacks (skip complete)"
  uv run python scripts/rebuild_roi_stacks.py "$DATA" --position 0 \
    --source-index "$SRC_DATA/roi/Pos0/index.json"

  echo "==> $(date -Iseconds) Spotiflow on all ROIs / times (channel 1)"
  uv run binding spotiflow "$DATA" \
    --roi-stacks --all-rois --all-times \
    -p 0 -c 1 \
    --model general --estimate-params \
    -o "$DATA/results/spotiflow_4s"

  echo "==> $(date -Iseconds) Filter spots"
  uv run binding filter-spots "$DATA/results/spotiflow_4s" \
    -o "$DATA/results/filtered_4s" \
    --min-intensity 1500 \
    --min-fwhm 2.0 --max-fwhm 8.0 \
    --min-probability 0.25

  echo "==> $(date -Iseconds) Per-frame spot counts"
  uv run binding spot-counts "$DATA/results/filtered_4s" \
    -o "$DATA/results" \
    -p 0 -c 1 \
    --time-interval "$TIME_INTERVAL" \
    --match-distance 5 \
    --per-frame
  mv -f "$DATA/results/spot_counts_position000_channel001.csv" \
    "$DATA/results/spot_counts_per_frame_position000_channel001.csv"

  echo "==> $(date -Iseconds) Cumulative spot counts"
  uv run binding spot-counts "$DATA/results/filtered_4s" \
    -o "$DATA/results" \
    -p 0 -c 1 \
    --time-interval "$TIME_INTERVAL" \
    --match-distance 5 \
    --cumulative
  mv -f "$DATA/results/spot_counts_position000_channel001.csv" \
    "$DATA/results/spot_counts_cumulative_position000_channel001.csv"

  echo "==> $(date -Iseconds) Particle tracking + merge events"
  uv run python scripts/track_spots.py "$DATA/results/filtered_4s" \
    -o "$DATA/results/tracking_4s" \
    --time-interval "$TIME_INTERVAL" \
    --max-distance 5 \
    --merge-distance 8

  echo "==> $(date -Iseconds) Done. Outputs under $DATA/results/"
} 2>&1 | tee -a "$LOG"
