#!/usr/bin/env bash
# Bootstrap the YOLO camera project with uv (bash version).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_VERSION="${PYTHON_VERSION:-3.10.12}"
VENV_PATH="${VENV_PATH:-yolo}"
REQ_FILE="${REQ_FILE:-${ROOT_DIR}/requirements.txt}"
TORCH_CUDA_VARIANT="${TORCH_CUDA_VARIANT:-cu124}"
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/${TORCH_CUDA_VARIANT}}"
INSTALL_CUDA_TORCH="${INSTALL_CUDA_TORCH:-auto}"

function log() {
    printf '[install_with_uv] %s\n' "$*"
}

function ensure_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        log "uv CLI not found. Install it from https://astral.sh/uv before running this script."
        exit 1
    fi
}

function ensure_python() {
    log "Ensuring Python ${PYTHON_VERSION} is available through uv..."
    uv python install "${PYTHON_VERSION}"
}

function ensure_venv() {
    if [ -d "${VENV_PATH}" ]; then
        log "Virtual environment '${VENV_PATH}' already exists. Skipping creation."
        return
    fi

    log "Creating virtual environment '${VENV_PATH}' with Python ${PYTHON_VERSION}..."
    uv venv "${VENV_PATH}" --python "${PYTHON_VERSION}"
}

function resolve_python_path() {
    if [ -f "${VENV_PATH}/bin/python" ]; then
        echo "${VENV_PATH}/bin/python"
    elif [ -f "${VENV_PATH}/Scripts/python.exe" ]; then
        echo "${VENV_PATH}/Scripts/python.exe"
    else
        log "Could not locate python executable in '${VENV_PATH}'."
        exit 1
    fi
}

function install_packages() {
    local python_path="$1"
    local requirements_file="$2"
    local extra_index=()
    local tmp_req=""

    if [ ! -f "${requirements_file}" ]; then
        log "Requirements file '${requirements_file}' not found."
        exit 1
    fi

    if should_install_cuda_torch; then
        tmp_req="$(mktemp)"
        generate_cuda_requirements "${requirements_file}" "${tmp_req}"
        requirements_file="${tmp_req}"
        extra_index=(--extra-index-url "${TORCH_CUDA_INDEX_URL}")
        log "CUDA-capable GPU detected. Using requirements with '${TORCH_CUDA_VARIANT}' wheels."
    else
        log "No CUDA-capable GPU detected or installation disabled. Using original requirements."
    fi

    log "Installing packages from ${requirements_file} with uv pip..."
    uv pip install --python "${python_path}" "${extra_index[@]}" -r "${requirements_file}"

    [ -n "${tmp_req}" ] && rm -f "${tmp_req}"
}

function should_install_cuda_torch() {
    case "${INSTALL_CUDA_TORCH}" in
        1|true|TRUE|yes|YES)
            return 0
            ;;
        0|false|FALSE|no|NO)
            return 1
            ;;
        auto|AUTO|"")
            if command -v nvidia-smi >/dev/null 2>&1; then
                return 0
            else
                return 1
            fi
            ;;
        *)
            log "Unknown INSTALL_CUDA_TORCH value '${INSTALL_CUDA_TORCH}', defaulting to 'auto'."
            if command -v nvidia-smi >/dev/null 2>&1; then
                return 0
            else
                return 1
            fi
            ;;
    esac
}

function generate_cuda_requirements() {
    local source_file="$1"
    local target_file="$2"

    : > "${target_file}"

    while IFS= read -r line || [ -n "${line}" ]; do
        line="${line%$'\r'}"
        case "${line}" in
            torch==*)
                line="$(convert_to_cuda_variant "${line}" torch)"
                ;;
            torchvision==*)
                line="$(convert_to_cuda_variant "${line}" torchvision)"
                ;;
            torchaudio==*)
                line="$(convert_to_cuda_variant "${line}" torchaudio)"
                ;;
        esac
        printf '%s\n' "${line}" >> "${target_file}"
    done < "${source_file}"
}

function convert_to_cuda_variant() {
    local line="$1"
    local package="$2"
    local version_part="${line#${package}==}"
    local base_version="${version_part%%+*}"
    printf '%s==%s+%s' "${package}" "${base_version}" "${TORCH_CUDA_VARIANT}"
}

ensure_uv
ensure_python
ensure_venv

PYTHON_BIN="$(resolve_python_path)"
install_packages "${PYTHON_BIN}" "${REQ_FILE}"

log "Environment ready."
if [ -f "${VENV_PATH}/bin/activate" ]; then
    log "Activate it with: source ${VENV_PATH}/bin/activate"
else
    log "Activate it with: source ${VENV_PATH}/Scripts/activate"
fi
