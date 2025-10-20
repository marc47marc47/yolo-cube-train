#!/usr/bin/env bash
# Download a pedestrian-only YOLO-format dataset derived from coco128.

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/reference/pedestrian}"
DATASET_URL="${DATASET_URL:-https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip}"
IMAGE_EXTENSIONS=("jpg" "jpeg" "png" "JPG" "JPEG" "PNG")

function log() {
    printf '[download_pedestrian_data] %s\n' "$*"
}

if [[ -z "${BASH_VERSION:-}" ]]; then
    log "Please run this script with bash (e.g., 'bash $0')."
    exit 1
fi

function require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        log "Required command '$1' not found in PATH."
        exit 1
    fi
}

require_cmd curl
require_cmd unzip
require_cmd awk

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

ZIP_PATH="${TMP_DIR}/dataset.zip"
log "Downloading dataset from ${DATASET_URL}..."
curl -L "${DATASET_URL}" -o "${ZIP_PATH}"

log "Unpacking dataset..."
unzip -q "${ZIP_PATH}" -d "${TMP_DIR}"

SRC_DIR="${TMP_DIR}/coco128"
if [ ! -d "${SRC_DIR}" ]; then
    log "Expected extracted directory '${SRC_DIR}' not found."
    exit 1
fi

SPLITS=("train2017" "val2017")

for split in "${SPLITS[@]}"; do
    LABEL_SRC="${SRC_DIR}/labels/${split}"
    IMAGE_SRC="${SRC_DIR}/images/${split}"

    if [ ! -d "${LABEL_SRC}" ] || [ ! -d "${IMAGE_SRC}" ]; then
        log "Missing expected split directories for '${split}'."
        exit 1
    fi

    TARGET_LABEL_DIR="${DATA_ROOT}/${split}/labels"
    TARGET_IMAGE_DIR="${DATA_ROOT}/${split}/images"
    mkdir -p "${TARGET_LABEL_DIR}" "${TARGET_IMAGE_DIR}"

    for label_file in "${LABEL_SRC}"/*.txt; do
        [ -e "${label_file}" ] || continue
        base="$(basename "${label_file}")"
        person_only="${TMP_DIR}/${base}"

        awk '$1 == 0 {print $0}' "${label_file}" > "${person_only}"

        if [ -s "${person_only}" ]; then
            cp "${person_only}" "${TARGET_LABEL_DIR}/${base}"
            image_base="${base%.txt}"
            image_file=""
            for ext in "${IMAGE_EXTENSIONS[@]}"; do
                candidate="${IMAGE_SRC}/${image_base}.${ext}"
                if [ -f "${candidate}" ]; then
                    image_file="${candidate}"
                    break
                fi
            done
            if [ -z "${image_file}" ]; then
                log "Warning: no matching image found for label '${label_file}'. Skipping."
                continue
            fi
            cp "${image_file}" "${TARGET_IMAGE_DIR}/"
        fi
    done
done

for split in "${SPLITS[@]}"; do
    count=$(find "${DATA_ROOT}/${split}/labels" -type f -name '*.txt' | wc -l | tr -d ' ')
    log "Prepared ${count} annotated samples for ${split}."
done

log "Pedestrian dataset ready under '${DATA_ROOT}'."
