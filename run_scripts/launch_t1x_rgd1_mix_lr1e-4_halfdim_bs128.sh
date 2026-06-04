#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

NNODES="${PET_NNODES:-${NNODES:-1}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23456}"

if [[ -n "${PET_NPROC_PER_NODE:-}" ]]; then
    NPROC_PER_NODE="${PET_NPROC_PER_NODE}"
elif [[ -n "${NPROC_PER_NODE:-}" ]]; then
    NPROC_PER_NODE="${NPROC_PER_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a CUDA_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC_PER_NODE="${#CUDA_DEVICES[@]}"
else
    NPROC_PER_NODE=4
fi

RUN_NAME="${RUN_NAME:-t1x_rgd1_mix_lr1e-4_halfdim_bs128}"
ALLOW_EXISTING_RUN_DIR="${ALLOW_EXISTING_RUN_DIR:-true}"
DISABLE_PROGRESS_BAR="${DISABLE_PROGRESS_BAR:-${SILENT:-true}}"

echo "[INFO] Launch config: nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[INFO] Run name: ${RUN_NAME}"
echo "[INFO] Allow existing run dir: ${ALLOW_EXISTING_RUN_DIR}"
echo "[INFO] Disable progress bar: ${DISABLE_PROGRESS_BAR}"
echo "[INFO] Model dim: hidden_channels=196 num_radial=96"

python3 -m torch.distributed.run \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    oa_reactdiff/trainer/train_ts1_rgd1_mix_ddp.py \
    --run_name "${RUN_NAME}" \
    --num_nodes "${NNODES}" \
    --lr 1e-4 \
    --bz 32 \
    --hidden_channels 196 \
    --num_radial 96 \
    --allow_existing_run_dir "${ALLOW_EXISTING_RUN_DIR}" \
    --disable_progress_bar "${DISABLE_PROGRESS_BAR}" \
    "$@"
