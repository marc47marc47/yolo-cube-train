#!/usr/bin/env bash
# Quick test harness for the quality-control project.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

function log() {
    printf '[test_basic] %s\n' "$*"
}

function resolve_python() {
    if [[ -x "${ROOT_DIR}/yolo/Scripts/python.exe" ]]; then
        echo "${ROOT_DIR}/yolo/Scripts/python.exe"
    elif [[ -x "${ROOT_DIR}/yolo/bin/python" ]]; then
        echo "${ROOT_DIR}/yolo/bin/python"
    else
        log "Could not locate virtualenv python under ${ROOT_DIR}/yolo."
        log "Run scripts/install_with_uv.sh before executing tests."
        exit 1
    fi
}

PYTHON_BIN="$(resolve_python)"
PYTEST_MARK="${PYTEST_MARK:-not slow}"

declare -a TARGET_PATHS
if [[ -n "${PYTEST_PATHS:-}" ]]; then
    # Allow callers to pass a space separated list via PYTEST_PATHS.
    read -r -a TARGET_PATHS <<< "${PYTEST_PATHS}"
else
    TARGET_PATHS=("${ROOT_DIR}/tests")
fi

log "Running pytest with marker '${PYTEST_MARK}'."
"${PYTHON_BIN}" -m pytest -m "${PYTEST_MARK}" "${TARGET_PATHS[@]}" "$@"

DEFAULT_WEIGHTS="${DEFAULT_WEIGHTS:-${ROOT_DIR}/artifacts/runs/qc/quality_control_v1/weights/best.pt}"
DATA_YAML="${DATA_YAML:-${ROOT_DIR}/data/quality_control/dataset.yaml}"
DEVICE="${DEVICE:-0}"

if [[ -f "${DEFAULT_WEIGHTS}" && -f "${DATA_YAML}" ]]; then
    log "Validating ${DEFAULT_WEIGHTS} against ${DATA_YAML}."
    "${PYTHON_BIN}" -m ultralytics val \
        model="${DEFAULT_WEIGHTS}" \
        data="${DATA_YAML}" \
        device="${DEVICE}" || log "Validation command exited with errors."
else
    log "Skipping validation because weights (${DEFAULT_WEIGHTS}) or dataset yaml (${DATA_YAML}) is missing."
fi

log "Tests completed."
