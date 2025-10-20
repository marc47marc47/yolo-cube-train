#!/usr/bin/env bash
# Helper script to launch a full training + validation cycle with sensible defaults.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

function log() {
    printf '[train_all] %s\n' "$*"
}

function resolve_python() {
    if [[ -x "${ROOT_DIR}/yolo/Scripts/python.exe" ]]; then
        echo "${ROOT_DIR}/yolo/Scripts/python.exe"
    elif [[ -x "${ROOT_DIR}/yolo/bin/python" ]]; then
        echo "${ROOT_DIR}/yolo/bin/python"
    else
        log "Could not find virtualenv under ${ROOT_DIR}/yolo. Run scripts/install_with_uv.sh first."
        exit 1
    fi
}

PYTHON_BIN="$(resolve_python)"

DATA_YAML="${DATA_YAML:-${ROOT_DIR}/data/quality_control/dataset.yaml}"
SCREENSHOTS_DIR="${SCREENSHOTS_DIR:-${ROOT_DIR}/artifacts/screenshots}"
DATA_MODE="${DATA_MODE:-quality}"
MODEL_PATH="${MODEL:-${ROOT_DIR}/yolov8n.pt}"
EPOCHS="${EPOCHS:-50}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
PROJECT="${PROJECT:-artifacts/runs/qc}"
RUN_NAME="${NAME:-quality_control_v1}"
rm -rf "${ROOT_DIR}/data/quality_control"

if [[ ! -f "${DATA_YAML}" ]]; then
    log "Dataset yaml not found at ${DATA_YAML}; preparing dataset from ${SCREENSHOTS_DIR}."
    "${PYTHON_BIN}" -m src.app prepare-data \
        --screenshots "${SCREENSHOTS_DIR}" \
        --output "${ROOT_DIR}/data/quality_control" \
        --mode "${DATA_MODE}"
    DATA_YAML="${ROOT_DIR}/data/quality_control/dataset.yaml"
fi

log "Run finished."
