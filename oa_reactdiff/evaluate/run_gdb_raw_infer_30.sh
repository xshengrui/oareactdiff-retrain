#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-oa_reactdiff/trainer/our_new_pretrained-ts1x-diff.ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/gdb_raw_rollouts}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TIMESTEPS="${TIMESTEPS:-150}"
NOISE_SCHEDULE="${NOISE_SCHEDULE:-polynomial_2}"
RESAMPLINGS="${RESAMPLINGS:-5}"
JUMP_LENGTH="${JUMP_LENGTH:-5}"
REPEATS="${REPEATS:-30}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_ATOMS="${MAX_ATOMS:--1}"
SINGLE_FRAG_ONLY="${SINGLE_FRAG_ONLY:-1}"

python oa_reactdiff/evaluate/infer_30.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset-path oa_reactdiff/data/GDB-10-rxn_raw.tar.gz \
  --output-dir "${OUTPUT_ROOT}/GDB-10" \
  --repeats "${REPEATS}" \
  --timesteps "${TIMESTEPS}" \
  --noise-schedule "${NOISE_SCHEDULE}" \
  --resamplings "${RESAMPLINGS}" \
  --jump-length "${JUMP_LENGTH}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-atoms "${MAX_ATOMS}" \
  --single-frag-only "${SINGLE_FRAG_ONLY}" \
  --device "${DEVICE}"

python oa_reactdiff/evaluate/infer_30.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset-path oa_reactdiff/data/GDB-17-rxn_raw.tar.gz \
  --output-dir "${OUTPUT_ROOT}/GDB-17" \
  --repeats "${REPEATS}" \
  --timesteps "${TIMESTEPS}" \
  --noise-schedule "${NOISE_SCHEDULE}" \
  --resamplings "${RESAMPLINGS}" \
  --jump-length "${JUMP_LENGTH}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-atoms "${MAX_ATOMS}" \
  --single-frag-only "${SINGLE_FRAG_ONLY}" \
  --device "${DEVICE}"
